#Authored by Jacob Patshkowski and Wilson Ibyishaka
#EENG 490 Eastern Washington University 

import websockets
import asyncio
import json
import time
import cv2
import numpy as np
import random


global locs
locs = ['wh1', 'wh2', 'wsh', 'sh1', 'sh2', 'seh', 'eh1',
        'eh2', 'enh', 'nh1', 'nh2', 'nwh',
        'ibw', 'wt1', 'wt2', 'wt3', 'wr', 'wl', 'obw',
        'ibs', 'st1', 'st2', 'st3', 'sr', 'sl', 'obs',
        'ibe', 'et1', 'et2', 'et3', 'er', 'el', 'obe',
        'ibn', 'nt1', 'nt2', 'nt3', 'nr', 'nl', 'obn']

global course
course = np.zeros((480, 480), dtype="uint8")



def start():
    global locs
    global course
    for i in range(len(locs)):
        draw_path(course, (i + 1, i + 1, i + 1), locs[i])


class Light:
    def __init__(self, ws, loc):
        self.ws = ws
        self.send = ''
        self.last = ''
        self.clock = time.time() - 10
        self.timer = 0
        self.name = loc
        self.times = 0

        self.ryg = [0,0,1]
        self.front = [1, 0]
        self.right = [1, 0]

        self.btn1 = 0
        self.btn2 = 0
        self.msg = ''


    async def recv_msg(self):
        while True:
            try:
                self.msg = await asyncio.wait_for(self.ws.recv(), .5)
                if self.msg == '1':
                    self.btn1 = 1
                elif self.msg == '2':
                    self.btn2 = 1
                await asyncio.sleep(0)

            except:
                pass

    def get_send(self):
        self.send = ''
        for i in range(len(self.ryg)):
            self.send = self.send + str(self.ryg[i])

        for j in range(len(self.right)):
            self.send = self.send + str(self.right[j])

        for k in range(len(self.front)):
            self.send = self.send + str(self.front[k])


class Car:
    def __init__(self, ws, color, name, lc, uc, lb, ub, power):
        self.send_dict = {'P': 60, 'D': 0, 'L':0, 'R':0}
        self.ws = ws
        self.rangec = [lc, uc]
        self.rangeb = [lb, ub]
        self.color = color

        self.speed = 0
        self.direct = 0
        self.name = name
        self.power = power
        self.lastpower = 0
        self.times = 0
        self.left = 0
        self.right = 0
        self.send_dict['P'] = self.power
        self.send_dict['D'] = self.direct
        self.send_dict['L'] = self.left
        self.send_dict['R'] = self.right

        self.lastdir = 0
        self.loc = ''
        self.newloc = ''
        self.prev = ''
        self.nxt = ''
        self.future = ''
        self.path = np.zeros((480, 480), dtype="uint8")
        self.collisioncourse = np.zeros((480, 480), dtype="uint8")

        self.x1, self.y1, self.w1, self.h1 = 0, 0, 0, 0
        self.x2, self.y2, self.w2, self.h2 = 0, 0, 0, 0
        self.ccy = 0
        self.ccx = 0
        self.cbx = 0
        self.cby = 0
        self.vx = 0
        self.vy = 0
        self.cbox = None
        self.bbox = None

        self.n = 0    # for paths with lights
        self.s = 0
        self.e = 0
        self.w = 0

    async def recv_msg(self):
        while True:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=0.001)
                my_dict = json.loads(msg)
                self.speed = my_dict["S"]

                await asyncio.sleep(0)

            except:
                pass

    def update_frame(self, frame):

        try:  #no box condition
            cv2.drawContours(frame, [self.cbox], 0, self.color, 3)
        except:
            pass

        cv2.putText(frame, (self.name), (self.x1 + 10 + self.w1, self.y1 + self.h1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    self.color, 2)
        cv2.putText(frame, (str(self.speed) + " c/s"), (self.x1 - 10, self.y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        self.draw_circles(frame, (0,0,0))
        cv2.circle(frame, (self.ccx, self.ccy), 3, self.color, -1)
        draw_path(frame, self.color, self.loc)
        draw_path(frame, self.color, self.nxt)


    async def update_path(self, frame):
        self.path = np.zeros((480, 480), dtype="uint8")
        self.collisioncourse = np.zeros((480, 480), dtype="uint8")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.rangec[0], self.rangec[1])

        m = cv2.inRange(hsv, self.rangeb[0], self.rangeb[1])
        result = cv2.bitwise_and(frame, frame, mask=m)
        _, mask1 = cv2.threshold(m, 254, 255, cv2.THRESH_BINARY)
        cnts, h = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            x = 20
            if cv2.contourArea(c) > x:
                z1 = [60,140]
                z2 = [60,140,340,420]
                z3 = [340,420,60,140]
                z4 = [340, 420]
                ((x, y),(_,_),_) = cv2.minAreaRect(c)
                offtrack1 = (z1[0] < x and x < z1[1] and z1[0] < y  and y < z1[1]) or (z2[0] < x and x < z2[1] and z2[2] < y and y < z2[3])
                offtrack2 = (z3[0] < x and x < z3[1] and z3[2] < y and y < z3[3]) or (z4[0] < x and x < z4[1] and z4[0] < y and y  < z4[1])

                if (not offtrack1 and not offtrack2):
                    cv2.circle(mask, (int(x), int(y)), 8, (255, 255, 255), -1)

        mask = mask
        cv2.imshow('mask',mask)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        _, mask1 = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        cnts, h = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            x = 500
            if cv2.contourArea(c) > x:
                self.x1, self.y1, self.w1, self.h1 = cv2.boundingRect(c)
                rect = cv2.minAreaRect(c)
                self.cbox = cv2.boxPoints(rect)
                self.cbox = np.int0(self.cbox)

                ((self.ccx, self.ccy),(_,_),_) = rect
                self.ccx = int(self.ccx)
                self.ccy = int(self.ccy)

                mask2 = np.zeros(m.shape, dtype="uint8")
                cv2.drawContours(mask2, [self.cbox], -1, color=(255, 255, 255), thickness=cv2.FILLED)

                mask = cv2.bitwise_and(m, mask2)

                result = cv2.bitwise_and(frame, frame, mask=mask)
                _, mask1 = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
                cnts, h = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for c in cnts:
                    x = 20
                    if cv2.contourArea(c) > x:
                        self.x2, self.y2, self.w2, self.h2 = cv2.boundingRect(c)
                        rect = cv2.minAreaRect(c)
                        self.bbox = cv2.boxPoints(rect)
                        self.bbox = np.int0(self.bbox)

                        ((self.cbx, self.cby), (_, _), _) = rect
                        self.cbx = int(self.cbx)
                        self.cby = int(self.cby)

                        vy = self.cby - self.ccy
                        vx = self.cbx - self.ccx
                        mag = np.sqrt(vy * vy + vx * vx)
                        if mag == 0:
                            vx = 0
                            vy = 0
                        else:
                            vy = vy / mag
                            vx = vx / mag
                        temp = vx
                        self.vx = -vy
                        self.vy = temp

            if self.loc != '':
                draw_path(self.path, (255, 255, 255), self.loc)
                temp0 = np.zeros(m.shape, dtype="uint8")
                temp1 = np.zeros(m.shape, dtype="uint8")
                draw_path(self.path, (255, 255, 255), self.loc)
                cv2.drawContours(temp1, [self.cbox], 0, self.color, -1)
                draw_path(temp0, (255, 255, 255), self.loc)
                self.collisioncourse = cv2.bitwise_and(temp0, temp1)

                if self.x2 != 0 and self.w2 != 0 and self.h2 != 0 and self.y2 != 0:  # current position
                    self.newloc = self.find_loc()

                if self.nxt == 'wt2':
                    if self.newloc == 'wl' or self.newloc == 'sl':
                        self.newloc = 'wt2'

                if self.newloc == self.nxt or self.nxt == '':
                    self.prev = self.loc
                    self.loc = self.nxt
                    self.nxt = self.future
                    self.future = self.nxt_loc(self.nxt)

                draw_path(self.path, (255, 255, 255), self.nxt)
                draw_path(self.collisioncourse, (255, 255, 255), self.nxt)
                draw_path(self.collisioncourse, (255, 255, 255), self.future)

                if self.prev != '':
                    draw_path(self.path, (255, 255, 255), self.prev)

                self.set_lights()
            else:  # start condition
                if self.x2 != 0 and self.w2 != 0 and self.h2 != 0 and self.y2 != 0:
                    self.loc = self.find_loc()

                    if self.loc != '':
                        self.nxt = self.nxt_loc(self.loc)
                        self.future = self.nxt_loc(self.nxt)

        self.get_dir()

        if (self.direct != 0 and self.direct != self.lastdir) or (self.power != self.lastpower) or (self.times % 100 == 0):
            await self.ws.send(json.dumps(self.send_dict, separators=(',', ':')))
            self.lastdir = self.direct
            self.lastpower = self.power
            self.times = 0
        self.times = self.times + 1
        await asyncio.sleep(0)

    def find_loc(self):
        global locs
        global course
        current = np.max(course[self.y2:self.y2 + self.h2, self.x2:self.x2 + self.w2])
        if current == 0:
            return ''
        else:
            return locs[current - 1]

    async def send_data(self):

        await self.ws.send(json.dumps(self.send_dict, separators=(',', ':')))
        await asyncio.sleep(0)

    def get_dir(self):
        d = [0, 0, 0]
        mask = np.zeros((480, 480), dtype="uint8")
        cv2.circle(mask, (int(self.cbx - 12 * self.vx), int(self.cby - 12 * self.vy)), 4, (255, 255, 255), -1)
        d[0] = int(np.max(cv2.bitwise_and(mask, self.path)) / 255)

        mask = np.zeros((480, 480), dtype="uint8")
        cv2.circle(mask, (int(self.cbx), int(self.cby)), 4, (255, 255, 255), -1)
        d[1] = int(np.max(cv2.bitwise_and(mask, self.path)) / 255)

        mask = np.zeros((480, 480), dtype="uint8")
        cv2.circle(mask, (int(self.cbx + 12 * self.vx), int(self.cby + 12 * self.vy)), 4, (255, 255, 255), -1)
        d[2] = int(np.max(cv2.bitwise_and(mask, self.path)) / 255)

        self.direct = 4 * d[0] + 2 * d[1] + 1 * d[2]
        self.send_dict['D'] = self.direct

    def set_lights(self):
        self.send_dict['L'] = 0
        self.left = 0
        self.send_dict['R'] = 0
        self.right = 0

        if self.loc[1] == 'l':
            self.send_dict['L'] = 1
            self.left = 1
        elif self.loc[1] == 'r':
            self.send_dict['R'] = 1
            self.right = 1
        elif self.loc[0] == 'o' or self.loc[0] == 'i' or self.loc[2] == 'h':
            self.send_dict['L'] = 1
            self.left = 1

        if self.right == 0 and self.left == 0:
            if self.nxt[1] == 'l':
                self.send_dict['L'] = 1
                self.left = 1
            elif self.nxt[1] == 'r':
                self.send_dict['R'] = 1
                self.right = 1
            elif self.nxt[0] == 'o' or self.nxt[0] == 'i' or self.nxt[2] == 'h':
                self.send_dict['L'] = 1
                self.left = 1

    def nxt_loc(self, loc):
        future = ''
        closed = {'w':self.w, 'n':self.n, 's':self.s, 'e':self.e}
        pair0 = {'w':'e', 'e':'w','n':'s', 's':'n'}
        pair1 = {'s':'w', 'n':'e','e':'s', 'w':'n'}

        if loc[1:] == 'h1':
            if (random.randint(1, 10) <= 7) and (closed[loc[0]] == 0):
                future = "ib" + loc[0]
            else:
                future = loc[0:2] + str(2)

        elif loc[0] == 'i':
            if random.randint(1, 10) <= 8 or closed[pair1[loc[2]]] == 0 or closed[pair0[loc[2]]] == 0:
                future = loc[-1] + "t1"
            else:
                future = loc[2] + 'r'
        elif loc[1:] == 't1':
            if random.randint(1, 10) >= 7 and closed[pair0[loc[0]]] == 0:
                future = loc[0] + "t2"
            else:
                future = loc[0] + 'l'

        elif loc[0] == 'o':
            d = {'w': 'e', 'e': 'w', 'n': 's', 's': 'n'}
            future = d[loc[2]] + 'h2'
        elif loc[1] == 'l':
            d = {'w': 's', 'e': 'n', 'n': 'w', 's': 'e'}
            future = d[loc[0]] + "t3"
        elif loc[1] == 'r':
            d = {'w': 'n', 'e': 's', 'n': 'e', 's': 'w'}
            future = "ob" + d[loc[0]]
        elif loc[1:] == 't2':
            future = loc[0] + "t3"
        elif loc[1:] == 't3':
            future = "ob" + loc[0]
        elif loc[-1] == 'h':
            future =loc[1] + "h1"
        elif loc[1:] == 'h2':
            d = {'w': 's', 'e': 'n', 'n': 'w', 's': 'e'}
            future = loc[0] + d[loc[0]] + "h"

        return future

    def draw_circles(self, frame, color):
        cv2.circle(frame, (self.cbx, self.cby), 4, color, -1)
        cv2.circle(frame, (int(self.cbx + 12 * self.vx), int(self.cby + 12 * self.vy)), 4, color, -1)
        cv2.circle(frame, (int(self.cbx - 12 * self.vx), int(self.cby - 12 * self.vy)), 4, color, -1)

def draw_path(frame, color, draw):
    if draw == 'wh1':
        cv2.line(frame, (50, 100), (50, 240), color, 5)
    elif draw == 'wh2':
        cv2.line(frame, (50, 240), (50, 380), color, 5)
    elif draw == 'eh1':

        cv2.line(frame, (430, 240), (430, 380), color, 5)
    elif draw == 'eh2':

        cv2.line(frame, (430, 100), (430, 240), color, 5)
    elif draw == 'nh1':

        cv2.line(frame, (240, 50), (380, 50), color, 5)
    elif draw == 'nh2':
        cv2.line(frame, (100, 50), (240, 50), color, 5)
    elif draw == 'sh1':

        cv2.line(frame, (100, 430), (240, 430), color, 5)
    elif draw == 'sh2':
        cv2.line(frame, (240, 430), (380, 430), color, 5)
    elif draw == 'nwh':

        cv2.ellipse(frame, (100, 100), (50, 50), 0, 180, 270, color, 5)
    elif draw == 'wsh':

        cv2.ellipse(frame, (100, 380), (50, 50), 180, 270, 360, color, 5)
    elif draw == 'seh':

        cv2.ellipse(frame, (380, 380), (50, 50), 0, 0, 90, color, 5)
    elif draw == 'enh':

        cv2.ellipse(frame, (380, 100), (50, 50), 180, 90, 180, color, 5)
    elif draw == 'ibs':

        cv2.ellipse(frame, (240, 380), (50, 50), 180, 180, 270, color, 5)
    elif draw == 'st3':

        cv2.line(frame, (290, 100), (290, 200), color, 5)
    elif draw == 'st2':
        cv2.line(frame, (290, 200), (290, 280), color, 5)
    elif draw == 'st1':
        cv2.line(frame, (290, 280), (290, 380), color, 5)
    elif draw == 'obs':
        cv2.ellipse(frame, (240, 100), (50, 50), 180, 90, 180, color, 5)
    elif draw == 'obn':

        cv2.ellipse(frame, (240, 380), (50, 50), 180, 270, 360, color, 5)
    elif draw == 'ibn':

        cv2.ellipse(frame, (240, 100), (50, 50), 180, 0, 90, color, 5)
    elif draw == 'nt1':

        cv2.line(frame, (190, 100), (190, 200), color, 5)
    elif draw == 'nt2':
        cv2.line(frame, (190, 200), (190, 280), color, 5)
    elif draw == 'nt3':
        cv2.line(frame, (190, 280), (190, 380), color, 5)
    elif draw == 'ibe':

        cv2.ellipse(frame, (380, 240), (50, 50), 0, 270, 360, color, 5)
    elif draw == 'obe':

        cv2.ellipse(frame, (100, 240), (50, 50), 180, 0, 90, color, 5)
    elif draw == 'et3':

        cv2.line(frame, (100, 190), (200, 190), color, 5)
    elif draw == 'et2':
        cv2.line(frame, (200, 190), (280, 190), color, 5)
    elif draw == 'et1':
        cv2.line(frame, (280, 190), (380, 190), color, 5)
    elif draw == 'obw':

        cv2.ellipse(frame, (380, 240), (50, 50), 0, 0, 90, color, 5)
    elif draw == 'ibw':

        cv2.ellipse(frame, (100, 240), (50, 50), 180, 270, 360, color, 5)
    elif draw == 'wt1':

        cv2.line(frame, (100, 290), (200, 290), color, 5)
    elif draw == 'wt2':
        cv2.line(frame, (200, 290), (280, 290), color, 5)
    elif draw == 'wt3':
        cv2.line(frame, (280, 290), (380, 290), color, 5)
    elif draw == 'wr':

        cv2.ellipse(frame, (100, 380), (90, 90), 180, 90, 180, color, 5)
    elif draw == 'sr':

        cv2.ellipse(frame, (380, 380), (90, 90), 180, 0, 90, color, 5)
    elif draw == 'er':

        cv2.ellipse(frame, (380, 100), (90, 90), 90, 0, 90, color, 5)
    elif draw == 'nr':

        cv2.ellipse(frame, (100, 100), (90, 90), 0, 0, 90, color, 5)
    elif draw == 'wl':

        cv2.ellipse(frame, (190, 190), (100, 100), 0, 0, 90, color, 5)
    elif draw == 'nl':

        cv2.ellipse(frame, (290, 190), (100, 100), 90, 0, 90, color, 5)
    elif draw == 'el':

        cv2.ellipse(frame, (290, 290), (100, 100), 180, 0, 90, color, 5)
    elif draw == 'sl':

        cv2.ellipse(frame, (190, 290), (100, 100), 0, 270, 360, color, 5)



def collision(cars):
    collisions = []
    power_dict = {}
    nums = []
    for i in range(len(cars)):
        power_dict[cars[i].name] = [50]
        for j in range(i+1, len(cars)):
            boom = cv2.bitwise_and(cars[i].collisioncourse, cars[j].collisioncourse)
            if np.max(boom) == 255:
                cv2.imshow('coll', boom)
                collisions.append([cars[i], cars[j], boom])
                nums.append(i)
                nums.append(j)

    for col in collisions:
        _, mask1 = cv2.threshold(col[2], 254, 255, cv2.THRESH_BINARY)
        cnts, h = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            x = 4
            if cv2.contourArea(c) > x:

                x, y, w, h = cv2.boundingRect(c)

                cx = int((2 * x + w) / 2)
                cy = int((2 * y + h) / 2)

                # 2 pix / cm - > cm / 2 pix
                L1_norm0 = (abs(cx - col[0].ccx) + abs(cy - col[0].ccy))/2   #in cms
                L1_norm1 = (abs(cx - col[1].ccx) + abs(cy - col[1].ccy))/2
                x = abs(col[0].ccx-col[1].ccx)/2
                y = abs(col[0].ccy-col[1].ccy)/2
                L2_norm = np.sqrt(x*x + y*y)

                s0 = .6*60
                s1 = .6*60
                t0 = L1_norm0/s0
                t1 = L1_norm1/s1

                if L1_norm0 > L1_norm1 or col[0].power < col[1].power:
                    if L1_norm0 - L1_norm1 < 30 or col[0].power < 60:
                        pow = round((1/0.6)*((2*(L1_norm0 - 30))/t0) - col[0].power)
                        if pow < 5:
                            power_dict[col[0].name].append(5)
                        else:
                            power_dict[col[0].name].append(pow)

                    if L2_norm < 80:
                        power_dict[col[0].name].append(5)

                elif L1_norm0 <= L1_norm1 or col[1].power < col[0].power:
                    if L1_norm1 - L1_norm0 < 30 or col[1].power < 60:
                        pow = round((1/0.6)*((2*(L1_norm1 - 30))/t1)-col[1].power)
                        if pow < 5:
                            power_dict[col[1].name].append(5)
                        else:
                            power_dict[col[1].name].append(pow)

                    if L2_norm < 80:
                        power_dict[col[1].name].append(5)

                if np.min(power_dict[col[0].name]) == 5 and np.min(power_dict[col[1].name]) == 5:
                    if L1_norm0 > L1_norm1:
                        power_dict[col[1].name] = [30]
                    else:
                        power_dict[col[0].name] = [30]

    for car in cars:
        car.send_dict['P'] = str(np.min(power_dict[car.name]))

async def update_lights(lights, cars):
    pairs1 = {'North': 3, 'South': 2, 'East': 1, 'West': 0}
    pairs2 = {'North': 2, 'South': 3, 'East': 0, 'West': 1}
    pairs0 = {'North': 1, 'South': 0, 'East': 2, 'West': 3}

    for light in lights:
        light.last = light.send

        if (light.msg == '1' or light.msg == '2'):
            light.clock = time.time()

        light.time = time.time() - light.clock
        if (light.btn1):   ## front
            if light.time < 10 and light.time >= 0:

                lights[pairs1[light.name]].ryg[1] = 1
                lights[pairs1[light.name]].ryg[2] = 0

                lights[pairs2[light.name]].ryg[1] = 1
                lights[pairs2[light.name]].ryg[2] = 0


                for car in cars:
                    if lights[pairs1[light.name]].name == 'West':
                        car.w = 1
                    elif lights[pairs1[light.name]].name == 'East':
                        car.e = 1
                    elif lights[pairs1[light.name]].name == 'North':
                        car.n = 1
                    elif lights[pairs1[light.name]].name == 'South':
                        car.s = 1




            elif light.time >= 10 and light.time < 20:
                if light.time < 12:
                    lights[pairs1[light.name]].ryg[0] = 1
                    lights[pairs1[light.name]].ryg[1] = 0

                    lights[pairs2[light.name]].ryg[0] = 1
                    lights[pairs2[light.name]].ryg[1] = 0

                elif light.time > 12 and light.time < 18:


                    light.front[0] = 0
                    light.front[1] = 1

                    lights[pairs1[light.name]].right[0] = 0
                    lights[pairs1[light.name]].right[1] = 1

                elif light.time > 18 and light.time < 18.5:


                    light.front[0] = 1
                    light.front[1] = 0

                    lights[pairs1[light.name]].right[0] = 1
                    lights[pairs1[light.name]].right[1] = 0

                elif light.time > 18.5 and light.time < 19:

                    light.front[0] = 0
                    light.front[1] = 1

                    lights[pairs1[light.name]].right[0] = 0
                    lights[pairs1[light.name]].right[1] = 1
                elif light.time > 19 and light.time < 19.5:

                    light.front[0] = 1
                    light.front[1] = 0

                    lights[pairs1[light.name]].right[0] = 1
                    lights[pairs1[light.name]].right[1] = 0
                elif light.time > 19.5 and light.time < 20:

                    light.front[0] = 0
                    light.front[1] = 1

                    lights[pairs1[light.name]].right[0] = 0
                    lights[pairs1[light.name]].right[1] = 1

            else:

                light.front[0] = 1
                light.front[1] = 0

                lights[pairs1[light.name]].right[0] = 1
                lights[pairs1[light.name]].right[1] = 0

                light.time = 0
                light.btn1 = 0

                for car in cars:
                    if lights[pairs1[light.name]].name == 'West':
                        car.w = 0
                    elif lights[pairs1[light.name]].name == 'East':
                        car.e = 0
                    elif lights[pairs1[light.name]].name == 'North':
                        car.n = 0
                    elif lights[pairs1[light.name]].name == 'South':
                        car.s = 0

        elif light.btn2:
            if light.time < 10 and light.time >= 0:

                light.ryg[1] = 1
                light.ryg[2] = 0

                lights[pairs0[light.name]].ryg[1] = 1
                lights[pairs0[light.name]].ryg[2] = 0



                for car in cars:
                    print(light.name)
                    if light.name == 'West':
                        car.w = 1
                    elif light.name == 'East':
                        car.e = 1
                    elif light.name == 'North':
                        car.n = 1
                    elif light.name == 'South':
                        car.s = 1


            elif light.time >= 10 and light.time < 20:
                if light.time < 12:
                    light.ryg[0] = 1
                    light.ryg[1] = 0

                    lights[pairs0[light.name]].ryg[0] = 1
                    lights[pairs0[light.name]].ryg[1] = 0

                elif light.time >= 12 and light.time < 18:


                    light.right[0] = 0
                    light.right[1] = 1

                    lights[pairs2[light.name]].front[0] = 0
                    lights[pairs2[light.name]].front[1] = 1

                elif light.time >= 18 and light.time < 18.5:


                    light.right[0] = 1
                    light.right[1] = 0

                    lights[pairs2[light.name]].front[0] = 1
                    lights[pairs2[light.name]].front[1] = 0

                elif light.time >= 18.5 and light.time < 19:

                    light.right[0] = 0
                    light.right[1] = 1

                    lights[pairs2[light.name]].front[0] = 0
                    lights[pairs2[light.name]].front[1] = 1
                elif light.time >= 19 and light.time < 19.5:

                    light.right[0] = 1
                    light.right[1] = 0

                    lights[pairs2[light.name]].front[0] = 1
                    lights[pairs2[light.name]].front[1] = 0
                elif light.time >= 19.5 and light.time < 20:


                    light.right[0] = 0
                    light.right[1] = 1

                    lights[pairs2[light.name]].front[0] = 0
                    lights[pairs2[light.name]].front[1] = 1
            else:

                light.right[0] = 1
                light.right[1] = 0

                lights[pairs2[light.name]].front[0] = 1
                lights[pairs2[light.name]].front[1] = 0

                light.time = 0
                light.btn2 = 0

                for car in cars:
                    print(light.name)
                    if light.name == 'West':
                        car.w = 0
                    elif light.name == 'East':
                        car.e = 0
                    elif light.name == 'North':
                        car.n = 0
                    elif light.name == 'South':
                        car.s = 0

        for light in lights:
            if(light.btn2 == 0 and lights[pairs2[light.name]].btn1 == 0 and lights[pairs0[light.name]].btn2 == 0
                    and lights[pairs1[light.name]].btn1 == 0):

                light.ryg[0] = 0
                light.ryg[1] = 0
                light.ryg[2] = 1




            if light.send != light.last or light.times % 40 == 0:
                light.get_send()
                await light.ws.send(light.send)
                light.times = 0

            light.times = light.times + 1
            await asyncio.sleep(0)

def bin2int(ip):
    my_int = 128*int(ip[0]) + 64*int(ip[1]) + 32*int(ip[2]) + 16*int(ip[3])
    my_int = my_int + 8*int(ip[4]) + 4*int(ip[5])+2*int(ip[6]) + int(ip[7])
    return my_int



lb = np.array([101, 179, 137])
ub = np.array([118, 255, 255])

lr = np.array([0,113,109])
ur = np.array([65,255,255])

lp = np.array([113, 54, 123])
up = np.array([150, 108, 210])

lwr = np.array([0, 0, 196])
uwr = np.array([179, 49, 255])

lwb = np.array([0, 0, 248])
uwb = np.array([112, 21, 255])

lwp = np.array([19, 0, 203])
uwp = np.array([155, 88, 255])

power = 60

binr = "00101010"
binb = "10010010"
binp = "00011101"

intr = bin2int(binr)
intb = bin2int(binb)
intp = bin2int(binp)

nl = 149
sl = 5
wl = 200
el = 189


async def main():
    start()

    ip_blue = "ws://192.168.137." + str(intb) + ":8765"
    ip_red = "ws://192.168.137." + str(intr) + ":8765"
    ip_purp = "ws://192.168.137." + str(intp) + ":8765"
    nip = "ws://192.168.137." + str(nl) + ":8765"
    sip = "ws://192.168.137." + str(sl) + ":8765"
    wip = "ws://192.168.137." + str(wl) + ":8765"
    eip = "ws://192.168.137." + str(el) + ":8765"

    async with websockets.connect(ip_blue) as ws1:
        async with websockets.connect(ip_red) as ws2:
            async with websockets.connect(ip_purp) as ws3:
                async with websockets.connect(nip) as wsn:
                    async with websockets.connect(sip) as wss:
                        async with websockets.connect(wip) as wsw:
                            async with websockets.connect(eip) as wse:
                                Blue = Car(ws1, (255, 0, 0), 'Blue', lb, ub, lwb, uwb, power)
                                Red = Car(ws2, (0, 0, 255), 'Red', lr, ur, lwb, uwb, power)
                                Purple = Car(ws3, (255, 0, 255), 'Purple', lp, up, lwb, uwb, power)
                                North = Light(wsn, 'North')
                                South = Light(wss, 'South')
                                West = Light(wsw, 'West')
                                East = Light(wse, 'East')

                                asyncio.create_task(Blue.recv_msg())
                                asyncio.create_task(Red.recv_msg())
                                asyncio.create_task(Purple.recv_msg())
                                asyncio.create_task(North.recv_msg())
                                asyncio.create_task(South.recv_msg())
                                asyncio.create_task(West.recv_msg())
                                asyncio.create_task(East.recv_msg())

                                vid = cv2.VideoCapture(0)
                                vid.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                                time.sleep(0.1)
                                while True:
                                    ret, frame = vid.read()
                                    wid = 640
                                    hig = 480
                                    sq = int((wid - hig) / 2)
                                    frame = frame[:, sq:-sq]
                                    await Blue.update_path(frame)
                                    await Red.update_path(frame)
                                    await Purple.update_path(frame)

                                    Blue.update_frame(frame)
                                    Red.update_frame(frame)
                                    Purple.update_frame(frame)

                                    collision([Blue,Red, Purple])

                                    try: #esp8266 need a bit more server code to operate more effecently
                                        await update_lights([North, South, West, East], [Blue,Red, Purple])
                                    except:
                                        print("Press q")

                                    cv2.imshow("frame", frame)
                                    if cv2.waitKey(1) & 0xFF == ord('q'):
                                        Blue.send_dict['P'] = 0
                                        Red.send_dict['P'] = 0
                                        Purple.send_dict['P'] = 0
                                        await Blue.send_data()
                                        await Red.send_data()
                                        await Purple.send_data()
                                        break


asyncio.get_event_loop().run_until_complete(main())
cv2.destroyAllWindows()
