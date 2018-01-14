#!/usr/bin/env python3

import time, random, math, sys, atexit
import subprocess, collections, multiprocessing, queue
import pygame, pigpio, evdev, numpy
from PIL import Image, ImageOps, ImageMath, ImageFilter, ImageDraw


class WiggleMode:
    """One vibration mode (a set of motors we expect to move the bot in a particular way)"""

    velocity_smoothing = 0.5

    def __init__(self, pwm):
        self.pwm = pwm
        self.velocity = numpy.array((0,0))
        self.timestamp = 0

    def update_velocity(self, vec):
        inv = 1.0 - self.velocity_smoothing
        self.velocity = self.velocity * self.velocity_smoothing + numpy.array(vec) * inv


class WiggleState:
    """Current state of the bot controller, saved on every timestep"""

    def __init__(self, mode_pwm):
        self.timestamp = None
        self.position = None
        self.velocity = None
        self.pwm_initial = 0.1
        self.current_mode = None
        self.last_mode = None
        self.mode_change_timestamp = 0
        self.modes = [WiggleMode(p) for p in mode_pwm]
        self.frame_counter = 0
        self.dt_histogram = [0] * 20

    def __repr__(self):
        return "<frame %06d, pos=%s, mode=%s pwmi=%.1f dthist=%s>" % (
            self.frame_counter, self.position, self.current_mode, self.pwm_initial, self.dt_histogram)


class WiggleBot:
    """Controls the robot motors and reads sensors, navigating based on a goal map.
       Runs in a separate process which tries to stay in sync with the tablet event rate.
       """

    minimum_speed = 1e-2
    pwm_initial_increment = 1e-4
    pwm_initial_decay = 1e-5
    pwm_acceleration = 1.012
    ray_samples = 32
    ray_length = 0.7
    ray_exponent = 1.8
    ray_edge_penalty = 2.0
    mode_revisit_period = 6.0
    minimum_mode_duration = 0.3

    def __init__(self):
        self.goal = None
        self.goal_queue = multiprocessing.Queue(2)
        self.state_queue = multiprocessing.Queue(4096)
        self.pi = pigpio.pi()
        self.tablet = TabletLoop(self.pi)
        self.motors = Motors(self.pi)
        self.state = WiggleState([
           (1, 0, 0),
           (0, 1, 0),
           (0, 0, 1),
        ])

    def start(self):
        self.process = multiprocessing.Process(target=self._proc)
        self.process.start()

    def _proc(self):
        # Motor control process
        while True:
            self.step()

    def step(self):
        # Main control loop is driven by tablet event timing; wait for a HID report
        self.tablet.read()
        position = self.tablet.rx.scaled_pos()
        timestamp = self.tablet.rx.timestamp

        # Update instantaneous position and velocity
        if self.state.position is not None:
            dt = timestamp - self.state.timestamp
            diff = position - self.state.position
            if dt > 0 and diff[0] != 0.0 and diff[1] != 0.0:
                hist = self.state.dt_histogram
                hist[min(len(hist)-1, int(dt * 2000))] += 1
                self.state.velocity = diff / dt
        self.state.position = position
        self.state.timestamp = timestamp
        self.state.frame_counter = self.tablet.frame_counter

        # If we haven't desynchronized recently and the tablet was in the same mode
        # at each end of this step, update the mode-specific smoothed velocity estimate.
        if (self.state.current_mode is not None
            and self.state.last_mode == self.state.current_mode
            and self.state.velocity is not None and self.tablet.is_synchronized()):
            mode = self.state.modes[self.state.current_mode]
            mode.timestamp = self.state.timestamp
            mode.update_velocity(self.state.velocity)
            self.update_initial_pwm(mode.velocity)
        self.state.last_mode = self.state.current_mode

        # Share the latest state
        try:
            self.state_queue.put_nowait(self.state)
        except queue.Full:
            print("State queue overflow")

        # Get the latest goal map
        try:
            while True:
                self.goal = self.goal_queue.get_nowait()
        except queue.Empty:
            pass

        # Change states maybe
        next_mode = self.choose_mode()
        if next_mode == self.state.current_mode:
            self.accelerate()
        else:
            self.set_mode(next_mode)

    def accelerate(self):
        self.motors.set([min(1.0, s * self.pwm_acceleration) for s in self.motors.speeds])

    def update_initial_pwm(self, velocity):
        # Adjust minimum PWM value to maintain minimum speed, using smoothed velocity
        if velocity.dot(velocity) < self.minimum_speed * self.minimum_speed:
            self.increase_minimum_pwm()
        else:
            self.state.pwm_initial = max(0.0, self.state.pwm_initial - self.pwm_initial_decay)

    def increase_minimum_pwm(self):
        self.state.pwm_initial = min(1.0, self.state.pwm_initial + self.pwm_initial_increment)
        self.set_to_minimum_pwm()

    def set_to_minimum_pwm(self):
        if self.state.current_mode is not None:
            pwm = self.state.modes[self.state.current_mode].pwm
            self.motors.set([p * self.state.pwm_initial for p in pwm])

    def set_mode(self, mode):
        self.state.current_mode = mode
        self.state.mode_change_timestamp = self.state.timestamp
        self.tablet.desync()
        self.set_to_minimum_pwm()

    def choose_mode(self):
        # Keep the current mode if it's too soon to change
        if self.state.timestamp - self.state.mode_change_timestamp < self.minimum_mode_duration:
            return self.state.current_mode

        # Refresh modes with old velocity estimates, starting with the oldest
        modes = self.state.modes
        oldest_mode = 0
        for i, mode in enumerate(modes):
            if mode.timestamp < modes[oldest_mode].timestamp:
                oldest_mode = i
        if self.state.timestamp - modes[oldest_mode].timestamp > self.mode_revisit_period:
            return oldest_mode

        # Evaluate rays to find the best choice
        scores = [self.evaluate_ray(mode.velocity) for mode in modes]
        best_mode = 0
        for i, score in enumerate(scores):
            if score > scores[best_mode]:
                best_mode = i
        return best_mode

    def evaluate_ray(self, vec):
        goal = self.goal
        if goal is None:
            return 0
        position = self.state.position
        if position is None:
            return 0
        if vec is None:
            return 0
        norm = numpy.linalg.norm(vec)
        if norm <= 0.0:
            return 0

        major_axis = numpy.max(self.goal.shape)
        origin = major_axis * position
        direction = vec / norm
        ray_length = major_axis * self.ray_length
        samples = numpy.arange(0,1,1/self.ray_samples).reshape((-1,1)) ** self.ray_exponent
        points = numpy.round(origin + samples * (ray_length * direction)).astype(int)
        clipped = numpy.clip(points, [0,0], numpy.array(goal.shape) - [1,1])
        total = numpy.sum(goal[clipped[:,0], clipped[:, 1]])
        edge_penalty = numpy.linalg.norm(clipped - points)
        return total - edge_penalty * self.ray_edge_penalty


class GreatArtist:
    """Image-based algorithms for goal determination and status output.
       The generated goal images are sent to the WiggleBot asynchronously.
       """
       
    frame_delay = 0.5

    def __init__(self, inspiration):
        self.bot = WiggleBot()
        self.display = Display()
        self.movie = VideoEncoder()

        self.inspiration = ImageOps.invert(Image.open(inspiration).convert('L'))
        self.progress = Image.new('L', self.inspiration.size, 0)
        self.debugview = Image.new('L', self.inspiration.size, 0)
        self.progress_draw = ImageDraw.Draw(self.progress)
        self.debugview_draw = ImageDraw.Draw(self.debugview)

        self.major_axis = max(*self.inspiration.size)
        self.large_blur = ImageFilter.GaussianBlur(self.major_axis/4)

        self.frame_counter = 0
        self.goal = None
        self.goal_timestamp = None
        self.mode_scores = None
        self.step_timestamp = None

    def run(self):
        movie_file = time.strftime('bot-%y%m%d-%H%M%S.mp4', time.localtime())
        self.movie.start(movie_file, self.inspiration.size)
        self.display.start(self.inspiration.size)
        self.bot.start()
        while True:
            self.step()

    def step(self):
        # Compute and send a new goal image
        sub = ImageMath.eval("convert(i - prog/2, 'L')", dict(i=self.inspiration, prog=self.progress))
        self.goal = ImageMath.eval("convert(a/2+b, 'L')", dict(a=sub, b=sub.filter(self.large_blur)))
        self.bot.goal_queue.put(numpy.asarray(self.goal))

        # Drain the queue of incoming states from the bot control process
        try:
            while True:
                prev_pos = self.bot.state.position
                self.bot.state = self.bot.state_queue.get_nowait()
                self.record_bot_travel(prev_pos, self.bot.state.position)
                self.draw_debug_vibration_modes()
        except queue.Empty:
            pass

        # Update the output images
        self.draw_debug_latest_position()
        status_im = Image.merge('RGB', (self.debugview, self.goal, ImageOps.invert(self.progress)))
        self.display.show(status_im)
        self.movie.encode(status_im)

        self.frame_counter += 1
        self.debugview.paste(im=0, box=(0, 0,)+self.debugview.size)

        print("[%06d] %r" % (self.frame_counter, self.bot.state))
        time.sleep(self.frame_delay)

    def record_bot_travel(self, from_pos, to_pos, distance_threshold=0.1):
        if from_pos is None or to_pos is None:
            return
        dist = to_pos - from_pos
        if dist.dot(dist) > distance_threshold * distance_threshold:
            # Ignore large jumps
            return

        s = self.major_axis
        self.progress_draw.line((s*from_pos[0], s*from_pos[1],
            s*to_pos[0], s*to_pos[1]), fill=255, width=1)

    def draw_debug_vibration_modes(self):
        modes = self.bot.state.modes
        current = self.bot.state.current_mode 

        for mode in modes:
            # Follow the bot with all available vectors
            self.draw_vibration_mode_line(mode, self.bot.state.position)

        if current is not None:

            # Oscilloscope trace for current mode
            x = self.bot.state.frame_counter % self.debugview.size[0]
            y = current
            self.debugview.paste(im=255, box=(x, y*4, x+1, (y+1)*4))

            # Oscilloscope trace for vectors; X is time, Y is mode
            grid = ((0.004 * self.bot.state.frame_counter) % 1.0, (2+current) * 0.05)
            self.draw_vibration_mode_line(modes[current], grid)

    def draw_debug_latest_position(self):
        pos = self.bot.state.position
        if pos is not None:
            s = self.major_axis
            width, height = self.debugview.size
            self.debugview.paste(im=255, box=(0, int(s*pos[1]), width, int(s*pos[1])+1))
            self.debugview.paste(im=255, box=(int(s*pos[0]), 0, int(s*pos[0])+1, height))

    def draw_vibration_mode_line(self, mode, from_pos, zoom=2.0, width=1):
        s = max(*self.debugview.size)
        if from_pos is not None and mode.velocity is not None:
            to_pos = (from_pos[0] + mode.velocity[0]*zoom, from_pos[1] + mode.velocity[1]*zoom)
            ipos = (int(s*from_pos[0]), int(s*from_pos[1]))
            self.debugview_draw.line((s*from_pos[0], s*from_pos[1], s*to_pos[0], s*to_pos[1]), fill=255, width=width)


class Motors:
    def __init__(self, pi, hz=200):
        self.pi = pi
        self.pins = (23, 24, 25)
        self.count = len(self.pins)
        for pin in self.pins:
            self.pi.set_PWM_frequency(pin, hz)
        self.off()
        atexit.register(self.off)

    def set(self, speeds):
        self.speeds = tuple(speeds)
        for speed, pin in zip(self.speeds, self.pins):
            self.pi.set_PWM_dutycycle(pin, max(0, min(255, int(255 * speed))))

    def off(self):
        self.set((0,) * self.count)


class TabletLoop:
    def __init__(self, pi):
        self.tx = TabletTx(pi)
        self.rx = TabletRx()
        self.toggle = False
        self.frame_counter = 0

    def read(self):
        self.rx.sync_read()
        self.frame_counter = self.frame_counter + 1
        self.update_pressure()

    def desync(self):
        if self.is_synchronized(): 
            self.toggle = not self.toggle
            self.update_pressure()

    def is_synchronized(self):
        return (self.rx.get_pressure() > 0.3) == self.toggle

    def update_pressure(self):
        anti_idleness_ramp = (self.frame_counter & 7) / 7 - 0.5
        self.tx.set_pressure(0.1 + 0.7 * float(self.toggle) + 0.05 * anti_idleness_ramp)


class TabletTx:
    def __init__(self, pi):
        self.pi = pi

    def set_pressure(self, p):
        self.set_hz(int(255000 + max(0.0, min(1.0, p)) * 9000))

    def set_hz(self, hz, duty=0.1):
        self.pi.hardware_PWM(18, hz, int(1e6 * duty))


class TabletRx:
    def __init__(self):
        ABS = evdev.ecodes.EV_ABS
        X, Y, P = evdev.ecodes.ABS_X, evdev.ecodes.ABS_Y, evdev.ecodes.ABS_PRESSURE
        for path in evdev.list_devices():
            dev = evdev.InputDevice(path)
            caps = dev.capabilities()
            if ABS in caps:
                ab = dict(caps[ABS])
                if X in ab and Y in ab and P in ab:
                    self.x, self.y, self.p = (ab[X], ab[Y], ab[P])
                    self.timestamp = 0
                    self.dev = dev
                    return
        raise IOError("Couldn't find the tablet (looking for an input device with X, Y, and Pressure)")

    def sync_read(self):
        SYN, ABS = evdev.ecodes.EV_SYN, evdev.ecodes.EV_ABS
        X, Y, P = evdev.ecodes.ABS_X, evdev.ecodes.ABS_Y, evdev.ecodes.ABS_PRESSURE
        for ev in self.dev.read_loop():
            if ev.type == SYN:
                self.timestamp = ev.timestamp()
                break
            if ev.type == ABS:
                if ev.code == X: self.x = self.x._replace(value=ev.value)
                if ev.code == Y: self.y = self.y._replace(value=ev.value)
                if ev.code == P: self.p = self.p._replace(value=ev.value)

    def scaled_pos(self):
        s = 1.0 / max(self.x.max - self.x.min, self.y.max - self.y.min)
        return numpy.array((s * (self.x.value - self.x.min), s * (self.y.value - self.y.min)))

    def get_pressure(self):
        return (self.p.value - self.p.min) / (self.p.max - self.p.min)


class Display:
    def start(self, size):
        self.size = size
        self.queue = multiprocessing.Queue(2)
        self.process = multiprocessing.Process(target=self._proc)
        self.process.start()

    def show(self, img):
        try:
            self.queue.put_nowait(img.tobytes())
        except queue.Full:
            pass

    def _proc(self):
        screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption('Wiggle Bot')
        while True:
            surf = pygame.image.fromstring(self.queue.get(), self.size, 'RGB')
            screen.blit(surf, dest=(0,0))
            pygame.display.update()


class VideoEncoder:
    def start(self, filename, size, fps=30, crf=15, zoom=2):
        self.filename = filename
        self.fps = fps
        self.crf = crf
        self.size = size
        filters = 'scale=%dx%d:flags=neighbor' % (size[0]*zoom, size[1]*zoom)
        self.proc = subprocess.Popen(['ffmpeg', '-y', '-pix_fmt', 'rgb24', '-f', 'rawvideo',
            '-s', '%dx%d' % self.size, '-r', str(self.fps),
            '-i', '-', '-crf', str(self.crf), '-vf', filters, self.filename], stdin=subprocess.PIPE)

    def encode(self, img):
        self.proc.stdin.write(img.tobytes())


if __name__ == "__main__":
    if len(sys.argv) == 2:
        GreatArtist(sys.argv[1]).run()
    else:
        sys.stderr.write('usage: %s <image file>\n' % sys.argv[0])


