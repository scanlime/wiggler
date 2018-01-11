#!/usr/bin/env python3

import time, random, math

import pigpio
import evdev
from evdev import ecodes
from PIL import Image, ImageDraw, ImageFilter, ImageMath, ImageOps


class Motors:
    def __init__(self, pi, hz=8000):
        self.pi = pi
        self.hz = hz
        self.pins = (23, 24, 25)

    def set(self, speeds):
        self.speeds = tuple(speeds)
        for speed, pin in zip(self.speeds, self.pins):
            self.pi.set_PWM_frequency(pin, self.hz)
            self.pi.set_PWM_dutycycle(pin, max(0, min(255, int(255 * speed))))

    def off(self):
        self.set((0,0,0))

    def random(self, pwm_level=0.8):
        d = random.randint(0, 2)
        mot = [ pwm_level * (d == i) for i in range(3) ]
        self.set(mot)

    def accelerate(self, rate=1.01):
        self.set([min(1.0, s * rate) for s in self.speeds])


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
                    self.x = absolute[ecodes.ABS_X]
                    self.y = absolute[ecodes.ABS_Y]
                    self.dev = dev
                    return
        raise IOError("No suitable tablet device found")

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

    def scaled_pos(self, image_size):
        x_size = self.x.max - self.x.min
        y_size = self.y.max - self.y.min
        scale = min(image_size[0] / x_size, image_size[1] / y_size)
        return (max(0, min(image_size[0]-1, (self.x.value - self.x.min) * scale)),
                max(0, min(image_size[1]-1, (self.y.value - self.y.min) * scale)))


class GreatArtist:
    def __init__(self, inspiration, eye_lag=40):
        self.inspiration = ImageOps.invert(Image.open(inspiration).convert('L'))
        self.progress = Image.new('L', self.inspiration.size, 0)
        self.draw = ImageDraw.Draw(self.progress)
        self.coarse_kernel = ImageFilter.GaussianBlur(64)
        self.fine_kernel = ImageFilter.GaussianBlur(4)
        self.pen_position = None
        self.counter = 0
        self.eye_feedback = 0
        self.pen_velocity = None

    def update(self, tablet_rx):
        next_pos = tablet_rx.scaled_pos(self.inspiration.size)
        if self.pen_position:
            self.draw.line(self.pen_position + next_pos, fill=255, width=1)
            self.pen_velocity = (next_pos[0] - self.pen_position[0], next_pos[1] - self.pen_position[1])

        sub = ImageMath.eval("convert(a-b*2/3, 'L')", dict(a=self.inspiration, b=self.progress))
        coarse = sub.filter(self.coarse_kernel).filter(self.coarse_kernel)
        self.eye = ImageMath.eval("convert((a+b)/2, 'L')", dict(a=sub, b=coarse)).filter(self.fine_kernel)

        self.eye.save("out/eye-%06d.png" % self.counter)
        self.progress.save("out/prog-%06d.png" % self.counter)
        self.counter = self.counter + 1
        self.pen_position = next_pos

    def should_change_direction(self, threshold=0.6):

        if not self.pen_velocity:
            return False

        current_score = self.evaluate_ray(self.pen_velocity)
        sampled_scores = self.evaluate_random_rays()
        print("current direction: %r   others: %r" % (current_score, sampled_scores))

        return current_score < sampled_scores[int(len(sampled_scores) * threshold)]

    def evaluate_random_rays(self, count=16):
        pen_speed = math.sqrt(math.pow(self.pen_velocity[0], 2) + math.pow(self.pen_velocity[1], 2))
        jitter = random.uniform(0, math.pi*2.0)
        rays = []
        for i in range(count):
            angle = i * math.pi * 2.0 / (count - 1.0)
            vec = (pen_speed * math.cos(angle), pen_speed * math.sin(angle))
            rays.append(self.evaluate_ray(vec))
        rays.sort()
        return rays

    def evaluate_ray(self, vec, step_length=1.5,  weight_step=0.2, edge_penalty=-5.0):
        """Score a ray starting at the current location, with the given per-frame velocity"""

        pos = self.pen_position
        total = 0
        weight = 1.0

        vec_len = math.sqrt(math.pow(vec[0], 2) + math.pow(vec[1], 2))
        if vec_len <= 0:
            return 0
        step_vec = (vec[0] * step_length / vec_len, vec[1] * step_length / vec_len)

        src = self.eye
        (width, height) = src.size

        while weight > 0.001:
            pos = (pos[0] + step_vec[0], pos[1] + step_vec[1])
            if pos[0] < 0 or pos[0] > width-1 or pos[1] < 0 or pos[1] > height-1:
                total += edge_penalty * weight
            else:
                total += src.getpixel(pos) * weight
            weight = weight * weight_step

        return total


def main():
    pi = pigpio.pi()
    tx = TabletTx(pi)
    rx = TabletRx()
    mot = Motors(pi)

    tx.set_idle()
    mot.random()

    a = GreatArtist("images/rng2.png")

    try:
        while True:
            print("Frame %d" % a.counter)
            rx.poll()
            a.update(rx)
            if a.should_change_direction():
                mot.random()
                print("change direction")
            else:
                mot.accelerate()
    finally:
        mot.off()

if __name__ == "__main__":
    main()

