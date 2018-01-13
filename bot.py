#!/usr/bin/env python3

import time, random, math, sys
import subprocess, collections, multiprocessing, queue

import pygame
import pigpio
import evdev
from evdev import ecodes
from PIL import Image, ImageDraw, ImageFilter, ImageMath, ImageOps


class Motors:
    def __init__(self, pi, hz=200):
        self.pi = pi
        self.pins = (23, 24, 25)
        self.count = len(self.pins)
        for pin in self.pins:
            self.pi.set_PWM_frequency(pin, hz)
        self.off()

    def set(self, speeds):
        self.speeds = tuple(speeds)
        for speed, pin in zip(self.speeds, self.pins):
            self.pi.set_PWM_dutycycle(pin, max(0, min(255, int(255 * speed))))

    def off(self):
        self.set((0,) * self.count)


class TabletTx:
    def __init__(self, pi):
        self.pi = pi
        self.set_idle()

    def set_idle(self):
        self.set_hz(255000)

    def set_hz(self, hz, duty=0.1):
        self.pi.hardware_PWM(18, hz, int(1e6 * duty))


class TabletRx:
    def __init__(self):
        for path in evdev.list_devices():
            dev = evdev.InputDevice(path)
            caps = dev.capabilities()
            if ecodes.EV_ABS in caps:
                absolute = dict(caps[ecodes.EV_ABS])
                if ecodes.ABS_X in absolute and ecodes.ABS_Y in absolute:
                    self.x, self.y = (absolute[ecodes.ABS_X], absolute[ecodes.ABS_Y])
                    self.dev = dev
                    return
        raise IOError("Couldn't find the tablet (looking for an input device with ABS_X and ABS_Y)")

    def poll(self):
        while True:
            event = self.dev.read_one()
            if not event:
                break
            if event.type == ecodes.EV_ABS:
                if event.code == ecodes.ABS_X:
                    self.x = self.x._replace(value=event.value)
                if event.code == ecodes.ABS_Y:
                    self.y = self.y._replace(value=event.value)

    def scaled_pos(self):
        x_size, y_size = (self.x.max - self.x.min, self.y.max - self.y.min)
        scale = min(1.0 / x_size, 1.0 / y_size)
        return ((self.x.value - self.x.min) * scale, (self.y.value - self.y.min) * scale)


class WiggleBot:
    pwm_initial_increment = 0.05
    pwm_initial_decay = 0.002
    pwm_acceleration = 1.012

    def __init__(self, use_combined_modes=True):
        self.pi = pigpio.pi()
        self.tablet_tx = TabletTx(self.pi)
        self.tablet_rx = TabletRx()
        self.motors = Motors(self.pi)

        self.position = None
        self.velocity = None
        self.frame_counter = 0
        self.pwm_initial = 0

        WiggleMode = collections.namedtuple('WiggleMode', ['pwm', 'velocity', 'timestamp'])
        self.vibration_modes = []
        if use_combined_modes:
            for mode_id in range(1, (1 << self.motors.count) - 1):
                pwm = [(mode_id >> m) & 1 for m in range(self.motors.count)]
                self.vibration_modes.append(WiggleMode(pwm=pwm, velocity=None, timestamp=None))
        else:
            for mode_id in range(self.motors.count):
                pwm = [(mode_id == m) for m in range(self.motors.count)]
                self.vibration_modes.append(WiggleMode(pwm=pwm, velocity=None, timestamp=None))
        self.change_mode(random.randrange(0, self.motors.count))

    def update(self):
        self.frame_counter += 1
        self.pwm_initial = max(0.0, self.pwm_initial - self.pwm_initial_decay)
        self.tablet_rx.poll()
        position = self.tablet_rx.scaled_pos()
        if self.position:
            self.velocity = (position[0] - self.position[0], position[1] - self.position[1])
            m = self.vibration_modes[self.current_mode]
            m = m._replace(velocity=self.velocity, timestamp=time.time())
            self.vibration_modes[self.current_mode] = m
        self.position = position

    def accelerate(self):
        self.motors.set([min(1.0, s * self.pwm_acceleration) for s in self.motors.speeds])

    def increase_minimum_pwm(self):
        self.pwm_initial = min(1.0, self.pwm_initial + self.pwm_initial_increment)
        self.change_mode(self.current_mode)

    def change_mode(self, mode):
        self.current_mode = mode
        pwm = self.vibration_modes[self.current_mode].pwm
        self.motors.set([p * self.pwm_initial for p in pwm])


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
    def start(self, filename, size, fps=30, crf=15):
        self.filename = filename
        self.fps = fps
        self.crf = crf
        self.size = size
        self.proc = subprocess.Popen([
            'ffmpeg', '-y',
            '-pix_fmt', 'rgb24',
            '-f', 'rawvideo',
            '-s', '%dx%d' % self.size,
            '-r', str(self.fps),
            '-i', '-',
            '-crf', str(self.crf),
            self.filename],
        stdin=subprocess.PIPE)

    def encode(self, img):
        self.proc.stdin.write(img.tobytes())


class GreatArtist:
    def __init__(self, inspiration):
        self.bot = WiggleBot()
        self.display = Display()
        self.movie = VideoEncoder()

        self.output_frame_count = 0
        self.inspiration = ImageOps.invert(Image.open(inspiration).convert('L'))
        self.progress = Image.new('L', self.inspiration.size, 0)
        self.debugview = Image.new('L', self.inspiration.size, 0)
        self.goal = None
        self.goal_timestamp = None
        self.mode_scores = None
        self.step_timestamp = None

        self.major_axis = max(*self.inspiration.size)
        self.large_blur = ImageFilter.GaussianBlur(self.major_axis/4)
        self.short_blur = ImageFilter.GaussianBlur(self.major_axis/150)

        movie_file = time.strftime('bot-%y%m%d-%H%M%S.m4v', time.localtime())
        self.movie.start(movie_file, self.inspiration.size)
        self.display.start(self.inspiration.size)

    def run(self):
        try:
            while True:
                self.step()
        finally:
            self.bot.motors.off()

    def time_step(self, step_duration):
        ts = time.time()
        if self.step_timestamp:
            delay_needed = step_duration - (ts - self.step_timestamp)
            if delay_needed > 0.001:
                time.sleep(delay_needed)
        self.step_timestamp = ts

    def step(self, goal_update_rate=1.0, min_step_duration=1/8, mode_change_delay=1/5):
        prev_position = self.bot.position
        self.bot.update()
        self.record_bot_travel(prev_position, self.bot.position)

        step_duration = min_step_duration
        if self.bot.velocity:
            if not self.goal:
                self.update_goal()
            else:
                next_mode = self.choose_mode()
                if next_mode == self.bot.current_mode:
                    self.bot.accelerate()
                else:
                    self.bot.change_mode(next_mode)
                    step_duration += mode_change_delay
                    if not self.goal_timestamp or time.time() - self.goal_timestamp > goal_update_rate:
                        self.update_goal()

        self.draw_debug_vibration_modes()
        self.time_step(step_duration)

        print("frame %06d, output %06d, mode=%d, scores=%r" % (
            self.bot.frame_counter, self.output_frame_count, self.bot.current_mode, self.mode_scores))

    def choose_mode(self, min_speed=2e-4):
        scores = list(map(self.evaluate_vibration_mode, range(len(self.bot.vibration_modes))))
        self.mode_scores = scores
        best_mode = 0
        for mode, score in enumerate(scores):
            info = self.bot.vibration_modes[mode]
            ts = info.timestamp
            vel = info.velocity or (0, 0)
            speed_squared = vel[0]*vel[0] + vel[1]*vel[1]
            if speed_squared < min_speed * min_speed:
                print("velocity bump, mode=%d speed=%s" % (mode, math.sqrt(speed_squared)))
                self.bot.increase_minimum_pwm()
                return mode
            elif score > scores[best_mode]:
                best_mode = mode
        return best_mode

    def record_bot_travel(self, from_pos, to_pos, distance_threshold=0.1):
        if not from_pos or not to_pos:
            return
        distance_squared = math.pow(to_pos[0] - from_pos[0], 2) + math.pow(to_pos[1] - from_pos[1], 2)
        if distance_squared > math.pow(distance_threshold, 2):
            return

        s = self.major_axis
        draw = ImageDraw.Draw(self.progress)
        draw.line((s*from_pos[0], s*from_pos[1], s*to_pos[0], s*to_pos[1]), fill=255, width=1)

    def draw_debug_vibration_modes(self):
        modes = self.bot.vibration_modes
        current = self.bot.current_mode 

        # Draw the mode we just chose, following the bot
        self.draw_vibration_mode_line(modes[current], self.bot.position)

        # Chart per-mode, along the bottom edge from the left
        for i, mode in enumerate(modes):
            grid = ((0.1 + i * 0.1), 0.1 + 0.02 * int((self.bot.frame_counter % 25)))
            self.draw_vibration_mode_line(mode, grid, width=(i == self.bot.current_mode)*4)

    def draw_debug_latest_position(self):
        pos = self.bot.position
        s = self.major_axis
        width, height = self.debugview.size
        self.debugview.paste(im=255, box=(0, int(s*pos[1]), width, int(s*pos[1])+1))
        self.debugview.paste(im=255, box=(int(s*pos[0]), 0, int(s*pos[0])+1, height))

    def draw_vibration_mode_line(self, mode, from_pos, zoom=40, width=1):
        draw = ImageDraw.Draw(self.debugview)
        s = max(*self.debugview.size)
        if from_pos and mode.velocity:
            to_pos = (from_pos[0] + mode.velocity[0]*zoom, from_pos[1] + mode.velocity[1]*zoom)
            ipos = (int(s*from_pos[0]), int(s*from_pos[1]))
            self.debugview.paste(im=200, box=(ipos[0]-1, ipos[1]-1, ipos[0]+2, ipos[1]+2))
            draw.line((s*from_pos[0], s*from_pos[1], s*to_pos[0], s*to_pos[1]), fill=255, width=width)

    def update_goal(self):
        sub = ImageMath.eval("convert(1 + i + i/10 - prog/2 - pblur/2, 'L')", dict(
            i=self.inspiration, prog=self.progress, pblur=self.progress.filter(self.short_blur)))
        long_distance_blur = sub.filter(self.large_blur)
        self.goal = ImageMath.eval("convert(a/2+b, 'L')", dict(a=sub, b=long_distance_blur))

        self.draw_debug_latest_position()

        status_im = Image.merge('RGB', (self.debugview, self.goal, ImageOps.invert(self.progress)))
        self.display.show(status_im)
        self.movie.encode(status_im)

        self.output_frame_count += 1
        self.goal_timestamp = time.time()
        self.debugview.paste(im=0, box=(0, 0,)+self.debugview.size)

    def sample_goal(self, pos, border=-1000, bias=128, gamma=1.8):
        size = self.goal.size
        to_pixels = max(*size)
        ipos = (int(pos[0] * to_pixels), int(pos[1] * to_pixels))
        if ipos[0] < 0 or ipos[0] > size[0]-1 or ipos[1] < 0 or ipos[1] > size[1]-1:
            return border

        self.debugview.putpixel(ipos, 128)
        p = self.goal.getpixel(ipos) - bias
        if p > 0:
            return math.pow(p, gamma)
        else:
            return -math.pow(-p, gamma)

    def evaluate_ray(self, vec, weight_multiple=0.1, length_multiple=1.3, num_samples=10, jitter=0.3):
        pos = self.bot.position
        total = 0
        weight = 1.0
        step_length = 1.0 / min(*self.goal.size)

        if not vec:
            return 0
        vec_len = math.sqrt(math.pow(vec[0], 2) + math.pow(vec[1], 2))
        if vec_len <= 0:
            return 0
        step_vec = (vec[0] * step_length / vec_len, vec[1] * step_length / vec_len)

        for i in range(num_samples):
            pos = (pos[0] + step_vec[0], pos[1] + step_vec[1])
            total += self.sample_goal(pos) * weight
            weight = weight * weight_multiple
            lma = random.uniform(1.0 - jitter, 1.0 + jitter) * length_multiple
            lmb = random.uniform(1.0 - jitter, 1.0 + jitter) * length_multiple
            step_vec = (step_vec[0] * lma, step_vec[1] * lmb)

        s = self.major_axis
        draw = ImageDraw.Draw(self.debugview)
        ipos = (int(s*pos[0]), int(s*pos[1]))
        label = "%.1f" % total
        textsize = draw.textsize(label)
        self.debugview.paste(im=0, box=(ipos[0], ipos[1], ipos[0]+textsize[0], ipos[1]+textsize[1]))
        draw.text(ipos, label, fill=180)

        return total

    def evaluate_vibration_mode(self, index, hysteresis=1.5, age_modifier=20.0):
        mode = self.bot.vibration_modes[index]
        score = self.evaluate_ray(mode.velocity)
        age = time.time() - (mode.timestamp or 0)
        score += age * age_modifier
        if index == self.bot.current_mode:
            score *= hysteresis
        return score

if __name__ == "__main__":
    if len(sys.argv) == 2:
        GreatArtist(sys.argv[1]).run()
    else:
        sys.stderr.write('usage: %s <image file>\n' % sys.argv[0])


