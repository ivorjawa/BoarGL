#!/usr/bin/env python2.7
from __future__ import division

from Queue import Full, Empty, Queue
#from multiprocessing import Process, Queue
import os
import atexit

import numpy as np
import pyaudio
import colorsys
import time


class Trawler(object):
    def shutdown(self):
        print "shutting down worker thread"
        #self.mosi.put(Kwitr())
        #self.fort_proc.join()
        self.stream.stop_stream()
        self.stream.close()
        self.paw.terminate()
        print "done"
    def __init__(self):
        self.miso = Queue()
        self.mosi = Queue()
        print "starting worker thread"
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        #paUint16 #paInt8
        CHANNELS = 1
        RATE = 44100 #sample rate

        self.paw = pyaudio.PyAudio()
        self.stream = self.paw.open(format=FORMAT,
                                    channels=CHANNELS,
                                    #input_device_index = 4, # rocketfish
                                    rate=RATE,
                                    input=True,
                                    stream_callback=self.callback,
                                    frames_per_buffer=CHUNK) #buffer
        #self.fort_proc = Process(target = fft_worker,
        #                         args=(self.mosi, self.miso))
        #self.fort_proc.start()
        atexit.register(self.shutdown)
        print "allegedly started worker"
    def fish(self):
        try:
            return self.miso.get(block=False)
        except Empty:
            return None
    def callback(self, in_data, frame_count, time_info, status):
        #print "callback data len: %s" % len(in_data)
        pixels = sample(in_data)
        if pixels != None:
            self.miso.put(pixels)
        time.sleep(1.0/30)
        #data = wf.readframes(frame_count)
        return ('', pyaudio.paContinue)


"""
you are close to the Shannon limit fs>2*sf, but matlab gives the good
answer.
the results of fft must be divided by N/2, where N is the length of data,
except for DC component (f=0) which must be divided by N.

http://www.mathworks.com/matlabcentral/newsreader/view_thread/25264
"""

"http://dsp.stackexchange.com/questions/16438/why-fft-does-not-retrieve-original-amplitude-when-increasing-signal-length"

def get_fft(y, fs):
    """ Get the FFT of a given signal and corresponding frequency bins.

    Parameters:
        y  - signal
        fs - sampling frequency
    Returns:
        (mag, freq) - tuple of spectrum magitude and corresponding frequencies
    """
    n  = len(y)      # Get the signal length
    dt = 1/float(fs) # Get time resolution

    fft_output = np.fft.rfft(y)     # Perform real fft
    rfreqs = np.fft.rfftfreq(n, dt) # Calculatel frequency bins
    fft_mag = np.abs(fft_output)    # Take only magnitude of spectrum

    # Normalize the amplitude by number of bins and multiply by 2
    # because we removed second half of spectrum above the Nyqist frequency
    # and energy must be preserved
    fft_mag = fft_mag * 2 / n

    return np.array(fft_mag), np.array(rfreqs)

def sample(data):
    try:
        #print "uhn."
        ndata = np.fromstring(data, dtype=np.int16)/32768
        mags, freqs = get_fft(ndata, 44100)
        mags *= 150 # amplify
        sumps = 120 # sample is 513, 0 is dc, most energy in first
        #           5000 hz, make an easy multiple of 120
        smags = mags[1:sumps+1]

        pixels = []
        hexamps = smags.reshape(30, 4) # reshape into a 30 x 4 array
        ###print "hexamps: %s" % hexamps
        for i, chunk in enumerate(hexamps):
            #avg = np.average(chunk)
            maxx = np.max(chunk)
            val = maxx
            val = min(1, val)
            val = max(.1, val)
            sat = 1.0
            hue = 1-(i/30.0)
            #print "h: %f s: %f v: %s" % (hue, sat, val),
            #pixel = np.array(colorsys.hsv_to_rgb(hue, sat, val))*256
            pixel = np.array(colorsys.hsv_to_rgb(hue, sat, val))
            #pixel = np.asarray(pixel, dtype=np.uint8)
            #print pixel
            pixels.append(pixel)
        return pixels
    except Exception, e:
        print "narf?: %s" % e
        return None


def Testors():
    import cv, cv2

    class Testor(object):
        def __init__(self):
            #self.cap = cv2.VideoCapture(0)
            #self.cap.set(3,640)
            #self.cap.set(4,480)
            self.timestamp = time.time()
            self.font = cv2.FONT_HERSHEY_SIMPLEX
            self.t = Trawler()
        def loop(self):
            while(1):
                p = self.t.fish()
                if(p):
                    self.showbars(p)
                if cv2.waitKey( 10) == 27:
                    break
                #time.sleep(1/30)
        def showbars(self, pixels):
            self.now = time.time()
            self.timedelt = self.now - self.timestamp
            self.timestamp = self.now
            im = np.zeros((480,640,3), np.uint8)

            im2 = np.copy(im)

            mids = np.array(range(32))*20 + 10
            for i in range(len(pixels)):
                xoff = i*20
                pixcol = np.array(pixels[i])
                barheit = int((np.sum(pixcol)/3.0) * 400)
                #print "i, pixcol, barheit:  %d: %s, %d" % (i, str(pixcol), barheit)
                # opencv is bgr  rgb
                #           210  012
                #pixcol.reverse()
                pc = pixcol*255
                cv2.rectangle(im2, (0+xoff, 0), (20+xoff, 479), (255,255,255), 1)
                cv2.rectangle(im2, (0+xoff, 40), (20+xoff, 40+barheit), pc, cv.CV_FILLED)
                #if( i < 30):
                #    p1 = (mids[i], 480-int(amp_values[i]*460+10))
                #    p2 = (mids[i+1], 480-int(amp_values[i+1]*460+10))
                #    cv2.line(im2, p1, p2, (0,255,0))

            #print "[pixel list of %d][%s]" % (len(pixels), str(pixels[0]))

            delt = "FR: %0.1f, px: %d, %s, %s" % (1/self.timedelt, len(pixels), str(im2.shape), str(self.timestamp))
            #delt = "^: %0.1f, px: %d" % (timedelt, len(pixels))
            cv2.putText( im2, delt, (30, 20), self.font, .4,
                         (0, 0, 255), 1, cv2.CV_AA)
            cv2.imshow('show bars', im2)
    t = Testor()
    t.loop()

def boring_text():
    t = Trawler()
    #time.sleep(1.0/30)
    for i in range(20):
        p = t.fish()
        if(p):
            print [(i, list(x)) for (i, x) in enumerate(p)]
            print
        time.sleep(1.0/30)

if __name__ == "__main__":
    Testors()

