import matplotlib.pyplot as plt
import matplotlib.image as mpng
import numpy as np
from scipy import signal
from zplane import zplane

# LABO

pi = np.pi
e = np.e


K = 1
z1 = np.complex(0, 0.8)
z2 = np.complex(0, -0.8)
p1 = 0.95*e**(np.complex(0, pi/8))
p2 = 0.95*e**(np.complex(0, -pi/8))

num = np.poly([z1, z2])
denum = np.poly([p1, p2])

w, Hw = signal.freqz(num, denum) # w = Fe?

def dB(x):
   return 20*np.log10(x)

def plot(y, x=None, title=""):
   plt.figure()
   if x:
      plt.plot(x, y)
   else:
      plt.plot(y)

   plt.title(title)

def prob1a():
   zplane(num, denum)


def prob1c():
   plt.figure()
   mod_Hw = np.abs(Hw)
   mod_Hw_dB = 20*np.log10(mod_Hw)
   plt.plot(w, mod_Hw_dB)
   plt.title('Réponse en fréquence Hw')
   plt.savefig('./figures/Hw_dB_plot')
   plt.xlabel('freq (rad/echan)')


def prob1d():
   impulse = np.array([0]*512 + [1] + [0]*511)
   yn = signal.lfilter(num, denum, impulse)

   plt.figure()
   plt.plot(yn)
   plt.title('y[n]')

   yn_DFT = np.fft.fft(yn)
   yn_DFT_dB = 20*np.log10(np.abs(yn_DFT))

   plt.figure()
   plt.plot(w, yn_DFT_dB[0:512]) #... devrait donner la même chose que 1c
   plt.title('Impulse response')


def prob1e():
   # On inverse le num et denum pour le nouveau filtre
   print()


#####################

def prob2a():

   z1 = e**np.complex(0, pi/16)
   z2 = e**np.complex(0, -pi/16)
   p1 = 0.95*e**np.complex(0, pi/16)
   p2 = 0.95*e**np.complex(0, -pi/16)

   num = np.poly([z1, z2])
   denum = np.poly([p1, p2])

   print(num)
   print(denum)

   # zplane(num, denum)

   w, Hw = signal.freqz(num, denum)
   plt.figure()
   plt.plot(w, Hw)
   plt.title('Filtre coupe bande')

   n = np.arange(0, 180*pi, 1)
   xn = np.sin(n*pi/16) + np.sin(n*pi/32)

   plt.figure()
   plt.plot(n, xn)
   plt.title('xn')

   yn2 = signal.lfilter(num, denum, xn)

   plt.figure()
   plt.plot(yn2)
   plt.title('yn')

   print()


####################

def prob3a():
   Fe = 48000
   wp = 2500/(Fe/2)
   ws = 3500/(Fe/2)
   gpass = 0.2
   gstop = 40

   print(wp, ws)

   order, wn = signal.buttord(wp, ws, gpass, gstop, False, Fe)
   num, denum = signal.butter(order, wn, 'lowpass', False, output='ba')

   print(order)

   w, Hw = signal.freqz(num, denum)

   plt.figure()
   plt.plot(w*Fe/(2*pi), dB(np.abs(Hw))) # En frequences
   plt.title('Butter')

   zeroes, poles, k = signal.butter(order, wn, 'lowpass', False, output='zpk')

   plt.figure()
   zplane(zeroes, poles)
   print(k)


def prob4():
   T = np.array([[2,0], [0, 0.5]])

   plt.gray()
   img = mpng.imread('./images_in/goldhill.png')
   plt.imshow(img)
   new_img = np.zeros((256, 1024))

   test = np.array([1, 2])
   new = np.matmul(T, test)
   print(new)

   for y in range(0, 512):
      for x in range(0, 512):
         newind = np.matmul(T, np.array([x, y]))

         new_img[int(newind[1])][int(newind[0])] = img[y][x]


   #plt.imshow(new_img)




def procedural_3():

   # Calculate by hand
   num = [0.16, 0.16]
   denum = [1, -0.668]
   w, Hw = signal.freqz(num, denum)

   plt.figure()
   plt.plot(Hw)


   # Compare with python function
   num, denum = signal.butter(1, 500, fs = 8000)  # Ordre 1, freq de coupure normalizé par Fe
   w, Hw2 = signal.freqz(num, denum)

   plt.figure()
   plt.plot(w, Hw2)

prob4()



plt.show()













