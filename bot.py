#!/usr/bin/env python3

import time, random, math, sys
from collections import namedtuple

import pigpio
import evdev
from evdev import ecodes
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageMath, ImageOps


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
    pwm_initial_increment = 0.01
    pwm_initial_decay = 0.001
    pwm_acceleration = 1.02

    def __init__(self):
        self.pi = pigpio.pi()
        self.tablet_tx = TabletTx(self.pi)
        self.tablet_rx = TabletRx()
        self.motors = Motors(self.pi)

        self.position = None
        self.velocity = None
        self.frame_counter = 0
        self.pwm_initial = 0

        WiggleMode = namedtuple('WiggleMode', ['pwm', 'velocity', 'timestamp'])
        self.vibration_modes = []
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
        print(self.pwm_initial)

    def change_mode(self, mode):
        self.current_mode = mode
        pwm = self.vibration_modes[self.current_mode].pwm
        self.motors.set([p * self.pwm_initial for p in pwm])


class GreatArtist:
    def __init__(self, bot, inspiration):
        self.bot = bot
        self.output_frame_count = 0
        self.font = ImageFont.truetype('DroidSansMono.ttf', 10)
        self.inspiration = ImageOps.invert(Image.open(inspiration).convert('L'))
        self.progress = Image.new('L', self.inspiration.size, 0)
        self.debugview = Image.new('L', self.inspiration.size, 0)
        self.goal = None
        self.goal_timestamp = None
        self.mode_scores = None
        self.step_timestamp = None
        self.sample_list = []
        self.large_blur = ImageFilter.GaussianBlur(max(*self.inspiration.size)//4)

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

    def step(self, goal_update_rate=1.0, min_step_duration=1/15, mode_change_delay=1/5):
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
        self.time_step(step_duration)

        print("frame %06d, output %06d, pwm=%r, scores=%r" % (
            self.bot.frame_counter, self.output_frame_count,
            self.bot.motors.speeds, self.mode_scores))

    def choose_mode(self, reevaluation_interval=2.5, min_speed=3e-5):
        scores = list(map(self.evaluate_vibration_mode, range(len(self.bot.vibration_modes))))
        self.mode_scores = scores
        best_mode = 0
        now = time.time()
        for mode, score in enumerate(scores):
            info = self.bot.vibration_modes[mode]
            ts = info.timestamp
            vel = info.velocity or (0, 0)
            speed_squared = vel[0]*vel[0] + vel[1]*vel[1]
            if speed_squared < min_speed * min_speed:
                print("velocity forcing mode=%d, speed=%s" % (mode, math.sqrt(speed_squared)))
                self.bot.increase_minimum_pwm()
                return mode
            if not ts or (now - ts) >= reevaluation_interval:
                print("timestamp forcing mode=%d, t=%s" % (mode, now - ts))
                return mode
            if score > scores[best_mode]: 
                best_mode = mode
        return best_mode

    def record_bot_travel(self, from_pos, to_pos, distance_threshold=0.1):
        if not from_pos or not to_pos:
            return
        distance_squared = math.pow(to_pos[0] - from_pos[0], 2) + math.pow(to_pos[1] - from_pos[1], 2)
        if distance_squared > math.pow(distance_threshold, 2):
            return

        s = max(*self.inspiration.size)
        draw = ImageDraw.Draw(self.progress)
        draw.line((s*from_pos[0], s*from_pos[1], s*to_pos[0], s*to_pos[1]), fill=255, width=1)

    def update_goal(self):
        sub = ImageMath.eval("convert(a-b, 'L')", dict(a=self.inspiration, b=self.progress))
        long_distance_blur = sub.filter(self.large_blur).filter(self.large_blur)
        self.goal = ImageMath.eval("convert(a+b, 'L')", dict(a=sub, b=long_distance_blur))

        s = max(*self.debugview.size)
        draw = ImageDraw.Draw(self.debugview)

        # Debug text
        velocities = ["v[%d] = %r" % (i, self.bot.vibration_modes[i].velocity)
                      for i  in range(len(self.bot.vibration_modes))]
        debug_text = "mode %d, frame %06d\npwm=%r\nscores=%r\n%s" % (
            self.bot.current_mode, self.bot.frame_counter,
            self.bot.motors.speeds, self.mode_scores,
            '\n'.join(velocities))
        draw.text((1,1), debug_text, font=self.font, fill=255)
       
        # Show (magnified) velocity estimates for each vibration mode
        for mode in self.bot.vibration_modes:
            from_pos = self.bot.position
            zoom = 100
            if from_pos and mode.velocity:
                to_pos = (from_pos[0] + mode.velocity[0]*zoom, from_pos[1] + mode.velocity[1]*zoom)
                w = 2 + 4 * (mode == self.bot.vibration_modes[self.bot.current_mode])
                draw.line((s*from_pos[0], s*from_pos[1], s*to_pos[0], s*to_pos[1]), fill=255, width=w)

        status_im = Image.merge('RGB', (self.debugview, self.goal, self.progress))
        status_im.save('out/%06d.png' % self.output_frame_count)
        self.output_frame_count += 1
        self.goal_timestamp = time.time()
        self.debugview.paste(im=0, box=(0, 0,)+self.debugview.size)

    def sample_goal(self, pos, border=0):
        size = self.goal.size
        to_pixels = max(*size)
        ipos = (int(pos[0] * to_pixels), int(pos[1] * to_pixels))
        if ipos[0] < 0 or ipos[0] > size[0]-1 or ipos[1] < 0 or ipos[1] > size[1]-1:
            return border

        self.debugview.putpixel(ipos, 128)
        return self.goal.getpixel(ipos)

    def evaluate_ray(self, vec, weight_multiple=0.2, length_multiple=1.1, num_samples=20):
        """Score a ray starting at the current location, with the given per-frame velocity"""

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
            step_vec = (step_vec[0] * length_multiple, step_vec[1] * length_multiple)

        return total

    def evaluate_rotated_ray(self, vec, degree_offset):
        angle = degree_offset * math.pi / 180.0
        s, c = (math.sin(angle), math.cos(angle))
        rotated = (vec[0]*c - vec[1]*s, vec[0]*s + vec[1]*c)
        return self.evaluate_ray(rotated)

    def evaluate_ray_bundle(self, vec):
        if not vec:
            return 0
        return (self.evaluate_ray(vec) +
                self.evaluate_rotated_ray(vec, -1) * 0.3 +
                self.evaluate_rotated_ray(vec, 1) * 0.3 +
                self.evaluate_rotated_ray(vec, -30) * 0.1 +
                self.evaluate_rotated_ray(vec, 30) * 0.1)               

    def evaluate_vibration_mode(self, index):
        mode = self.bot.vibration_modes[index]
        return self.evaluate_ray_bundle(mode.velocity)


def main(input_file):
    GreatArtist(WiggleBot(), input_file).run()

if __name__ == "__main__":
    main(sys.argv[1])

