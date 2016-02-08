#import hid
import atexit

js = None
def init_js():
    return # nerf it
    def cleanup():
        global js
        print "shutting down joystick"
        js.close()
        print "done"
    global js
    try:
        js = hid.device()
        js.open(1356, 616)
        atexit.register(cleanup)
        print "joystick initialized"
    except Exception, e:
        js = None
        raise

def trans_js(reading):
    #scale = 1/m.exp(2)
    #m.exp(2*((1/128)*p))*scale
    # fixme should make this logrithmic
    if (reading >= 133):
        z = reading -133
        return (reading-133) / 122.0
    elif (reading <= 122):
        z = 122 - reading
        return -((122-reading)/122.0)
    else:
        return 0

def trans_button(reading):
    "translate 8-bit unsigned to 0..1.0"
    if(reading > 10):
        return reading/255.0
    else:
        return 0

inkey = None
def set_key(key):
    global inkey
    #print "set key %s: %d" % (key, ord(key))
    inkey = key

def read_js(cam):
    global js, inkey
    rotax_x = 0
    rotax_y = 0
    moveax_x = 0
    moveax_y = 0
    rolax_left = 0
    rolax_right = 0
    d_up = 0
    d_down = 0
    d_left = 0
    d_right = 0

    if js != None:
        data = js.read(50)
        #print "data len: %s" % len(data)
        #left stick
        rotax_x = trans_js(data[6])
        rotax_y = trans_js(data[7])

        #right stick
        moveax_x = trans_js(data[8])
        moveax_y = trans_js(data[9])

        rolax_left = trans_js(data[18])
        rolax_right = trans_js(data[19])

        d_up = trans_button(data[14])
        d_right = trans_button(data[15])
        d_down = trans_button(data[16])
        d_left = trans_button(data[17])

    #print "js inkey: %s" % inkey
    if inkey == 'w':
        #print "for"
        cam.trans_z(.3)
    elif inkey == 'a':
        #print "left"
        cam.trans_x(.3)
    elif inkey == 's':
        #print "back"
        cam.trans_z(-.3)
    elif inkey == 'd':
        #print "right"
        cam.trans_x(-.3)
    elif inkey == '8':
        # up
        cam.trans_y(.3)
    elif inkey == '2':
        # down
        cam.trans_y(-.3)
    else:
        pass
    inkey = None

    if(moveax_x != 0):
        cam.yaw(moveax_x * -.05)
    if(moveax_y != 0):
        #cam.trans_y(rotax_y * -2)
        cam.pitch(moveax_y * .05)
    if(rolax_left != 0):
        cam.roll(rolax_left * -.05)
    if(rolax_right != 0):
        cam.roll(rolax_right * .05)

    if(rotax_x != 0):
        cam.trans_x(rotax_x * -.3)
    if(rotax_y != 0):
        cam.trans_z(rotax_y * -.3)
    if(d_up != 0):
        cam.trans_y(d_up * .3)
    if(d_down != 0):
        cam.trans_y(d_down * -.3)

    # zoom crap, needs more variables
    #if(d_left != 0):
    #    FOV += d_left * .3
    #    if(FOV >= 179.9):
    #        FOV = 179.9
    #    reshape(win_width, win_height)
    #if(d_right != 0):
    #    FOV -= d_right * .3
    #    if(FOV <= 0):
    #        FOV = .1
    #    reshape(win_width, win_height)


    ax = "Mx: %0.2f  My: %0.2f  Rx: %0.2f  Ry: %0.2f" % (moveax_x, moveax_y, rotax_x, rotax_y)
    return ax
    #except Exception, e:
    #    return str(e)
