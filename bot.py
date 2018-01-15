#!/usr/bin/env python3

import time, random, math, sys, queue
import subprocess, collections, multiprocessing
import pygame, pigpio, evdev, numpy
from PIL import Image, ImageOps, ImageMath, ImageFilter, ImageDraw


class WiggleBot:
    """Controls the robot motors and reads sensors, navigating based on a goal map.
       Runs in a separate process which tries to stay in sync with the tablet event rate.
       """

    minimum_speed = 0.04
    velocity_smoothing = 0.94
    pwm_initial_startup = 0.25
    pwm_initial_increment = 0
    pwm_initial_decay = 0
    pwm_acceleration = 1.012
    ray_samples = 32
    ray_length = 0.7
    ray_shape_exponent = 6.0
    ray_edge_penalty = 0.1
    ray_sample_bias = 20
    ray_sample_exponent = 2.0
    mode_revisit_period = 4.0
    minimum_mode_duration = 0.15

    mode_pwm = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]

    def __init__(self):
        self.state = WiggleState(self.mode_pwm, self.pwm_initial_startup)

    def start(self):
        self.goal_queue = multiprocessing.Queue(2)
        self.state_rx, self.state_tx = multiprocessing.Pipe(False)
        self.process = multiprocessing.Process(target=self._proc)
        self.process.start()

    def _proc(self):
        # Motor control process

        self.goal = None
        self.pi = pigpio.pi()
        self.tablet = TabletLoop(self.pi)
        self.motors = Motors(self.pi)

        try:
            while True:
                self.step()
        finally:
            self.shutdown()
            
    def shutdown(self):
        # self.pi might be locked
        pi = pigpio.pi()
        Motors(pi).off()
        TabletTx(pi).off()

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
            mode.update_velocity(self.state.velocity, self.velocity_smoothing)
            if mode.timestamp - self.state.mode_change_timestamp > self.minimum_mode_duration:
                self.update_initial_pwm(mode.velocity)
        self.state.last_mode = self.state.current_mode

        # Share the latest state
        self.state_tx.send(self.state)

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
        # Adjust minimum PWM value to maintain minimum speed.
        # Called with smoothed velocity after a mode is old enough to switch.
        
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
        self.state.ray_debug_info = []

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
        scores = []
        best_mode = 0
        for mode in modes:
            ray_info = self.evaluate_ray(mode.velocity)
            self.state.ray_debug_info.append(ray_info)
            scores.append(ray_info[0])
        for i, score in enumerate(scores):
            if score > scores[best_mode]:
                best_mode = i
        return best_mode

    def evaluate_ray(self, vec):
        goal = self.goal
        if goal is None:
            return (0, dict(error='no_goal_yet'))
        position = self.state.position
        if position is None:
            return (0, dict(error='no_position_yet'))
        if vec is None:
            return (0, dict(error='no_input_yet'))
        norm = numpy.linalg.norm(vec)
        if norm <= 0.0:
            return (0, dict(error='input_vec_zero'))

        height, width = self.goal.shape
        major_axis = numpy.max(self.goal.shape)

        origin = major_axis * position
        direction = vec / norm
        ray_length = major_axis * self.ray_length
        sample_locs = numpy.arange(0,1,1/self.ray_samples).reshape((-1,1)) ** self.ray_shape_exponent
        points = numpy.round(origin + sample_locs * (ray_length * direction)).astype(int)
        clipped = numpy.clip(points, (0,0), (width-1, height-1))
        samples = self.ray_sample_bias + goal[clipped[:,1], clipped[:,0]] ** self.ray_sample_exponent
        total = numpy.linalg.norm(samples)
        edge_penalty = numpy.linalg.norm(clipped - points)
        score = total - edge_penalty * self.ray_edge_penalty
        return (score, dict(clipped_points=clipped))


class WiggleMode:
    """One vibration mode (a set of motors we expect to move the bot in a particular way)"""

    def __init__(self, pwm):
        self.pwm = pwm
        self.velocity = numpy.array((0,0))
        self.timestamp = 0

    def update_velocity(self, vec, smoothing):
        self.velocity = self.velocity * smoothing + numpy.array(vec) * (1-smoothing)
    

class WiggleState:
    """Current state of the bot controller, saved on every timestep"""

    def __init__(self, mode_pwm, pwm_initial):
        self.timestamp = None
        self.position = None
        self.velocity = None
        self.pwm_initial = pwm_initial
        self.ray_debug_info = []
        self.current_mode = None
        self.last_mode = None
        self.mode_change_timestamp = 0
        self.modes = [WiggleMode(p) for p in mode_pwm]
        self.frame_counter = 0
        self.dt_histogram = [0] * 20

    def __repr__(self):
        return "<frame %06d, pos=%s, mode=%s pwmi=%.6f dthist=%s>" % (
            self.frame_counter, self.position, self.current_mode, self.pwm_initial, self.dt_histogram)


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
        self.large_blur = ImageFilter.GaussianBlur(96)

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
        while self.bot.state_rx.poll():
            prev_pos = self.bot.state.position
            self.bot.state = self.bot.state_rx.recv()
            self.record_bot_travel(prev_pos, self.bot.state.position)
            self.draw_per_state_debug()

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

    def draw_per_state_debug(self):
        modes = self.bot.state.modes
        current = self.bot.state.current_mode 

        for mode in modes:
            # Follow the bot with all available vectors
            self.draw_vibration_mode_line(mode, self.bot.state.position)

        # Trace current frame, along bottom of screen
        w, h = self.debugview.size
        frame_x = self.bot.state.frame_counter % w
        self.debugview.paste(im=255, box=(frame_x, h-4, frame_x+1, h))

        if current is not None:
            # Oscilloscope trace for current mode
            self.debugview.paste(im=255, box=(frame_x, current*4, frame_x+1, (current+1)*4))

            # Oscilloscope trace for vectors; X is time, Y is mode
            grid = ((0.004 * self.bot.state.frame_counter) % 1.0, (2+current) * 0.05)
            self.draw_vibration_mode_line(modes[current], grid)

        # Per-ray debugging from this state
        for score, ray_info in self.bot.state.ray_debug_info:
            for pt in ray_info.get('clipped_points', []):
                self.debugview.putpixel(pt, 180)

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
        self.update_pressure()

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
    idle_hz = 255000

    def __init__(self, pi):
        self.pi = pi

    def set_pressure(self, p):
        self.set_hz(int(self.idle_hz + max(0.0, min(1.0, p)) * 9000))

    def set_hz(self, hz, duty=0.1):
        self.pi.hardware_PWM(18, hz, int(1e6 * duty))

    def off(self):
        self.set_hz(self.idle_hz, 0)


class TabletRx:
    """Receive low-level tablet events via the Linux Event subsystem"""

    def __init__(self):
        ABS = evdev.ecodes.EV_ABS
        X, Y, P = evdev.ecodes.ABS_X, evdev.ecodes.ABS_Y, evdev.ecodes.ABS_PRESSURE
        for path in evdev.list_devices():
            dev = evdev.InputDevice(path)
            caps = dev.capabilities()
            if ABS not in caps:
                continue
            ab = dict(caps[ABS])
            if X in ab and Y in ab and P in ab:
                self.x, self.y, self.p = (ab[X], ab[Y], ab[P])
                self.timestamp = 0
                self.dev = dev
                print("Using device as tablet:", dev)
                dev.grab()
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
    zoom = 2

    def start(self, size):
        self.size = numpy.array(size)
        self.queue = multiprocessing.Queue(2)
        self.process = multiprocessing.Process(target=self._proc)
        self.process.start()

    def show(self, img):
        try:
            self.queue.put_nowait(img.tobytes())
        except queue.Full:
            pass

    def _proc(self):
        screen = pygame.display.set_mode(self.size * self.zoom)
        pygame.display.set_caption('Wiggle Bot')
        while True:
            surf = pygame.image.fromstring(self.queue.get(), self.size, 'RGB')
            pygame.transform.scale(surf.convert(screen), screen.get_size(), screen)
            pygame.display.update()


class VideoEncoder:
    def start(self, filename, size, fps=30, crf=15, zoom=2, vid_size=(1280,720)):
        self.filename = filename
        self.fps = fps
        self.crf = crf
        self.size = size
        vf_scale = 'scale=%dx%d:flags=neighbor' % (size[0]*zoom, size[1]*zoom)
        vf_pad = 'pad=%d:%d:(ow-iw)/2:(oh-ih)/2' % vid_size
        filters = (vf_scale, vf_pad)
        self.proc = subprocess.Popen([
            'ffmpeg', '-y', '-pix_fmt', 'rgb24', '-f', 'rawvideo',
            '-s', '%dx%d' % self.size, '-r', str(self.fps),
            '-i', '-', '-crf', str(self.crf),
            '-vf', ','.join(filters), self.filename],
            stdin=subprocess.PIPE)

    def encode(self, img):
        self.proc.stdin.write(img.tobytes())


if __name__ == "__main__":
    if len(sys.argv) == 2:
        GreatArtist(sys.argv[1]).run()
    else:
        sys.stderr.write('usage: %s <image file>\n' % sys.argv[0])


