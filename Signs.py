#!/usr/bin/python
from colorsys import hsv_to_rgb
from fileinput import FileInput
from gc import collect
from glob import glob
from itertools import chain, dropwhile
from logging import basicConfig, getLogger, Formatter, DEBUG
from math import cos, floor, log, pi, radians, sin
from os import makedirs
from os.path import dirname, isdir, join
from random import choice, randint, random, seed, uniform
from signal import signal, SIGTERM
from sys import argv, exit # pylint: disable=W0622
from time import time

print
print "               Wall of Signs v1.82"
print "for 'House where the World Sounds...' Russian LARP"
print "http://house-where.livejournal.com/275.html"
print "http://house-where.livejournal.com/3335.html"
print "http://house-where.livejournal.com/68372.html"
print "Press ESC or Alt-F4 to quit, Alt-Enter to toggle fullscreen, Space for pause"
print

basicConfig(level = DEBUG, stream = None, format = '%(asctime)s %(name)s %(levelname)s %(message).47s', datefmt = '%H:%M:%S') # Cut too long OpenGL exception messages

try:
    from Image import frombuffer as imageFromBuffer, open as imageOpen
except ImportError, ex:
    raise ImportError("%s: %s\n\nPlease install PIL v1.1.7 or later: http://pythonware.com/products/pil/\n" % (ex.__class__.__name__, ex))

try:
    import numpy # pylint: disable=W0611
except ImportError, ex:
    raise ImportError("%s: %s\n\nPlease install NumPy v1.6.2 or later: http://numpy.scipy.org\n" % (ex.__class__.__name__, ex))

try:
    from pygame import init as pyGameInit
    from pygame import Rect, Surface
    from pygame.surfarray import array2d, pixels2d
except ImportError, ex:
    raise ImportError("%s: %s\n\nPlease install PyGame v1.9.1 or later: http://pygame.org\n" % (ex.__class__.__name__, ex))

try:
    from pygame.font import SysFont
except ImportError, ex:
    raise ImportError("%s: %s\n\npygame.font is not available, can't continue, exiting\n" % (ex.__class__.__name__, ex))

try:
    from OpenGL.GL import glBindTexture, glBegin, glBlendFunc, glClear, glClearColor, glColor3f
    from OpenGL.GL import glDeleteTextures, glDepthMask, glDisable, glEnable, glEnd
    from OpenGL.GL import glGenTextures, glGetError, glLoadIdentity, glMatrixMode
    from OpenGL.GL import glTexCoord2f, glTexParameteri, glVertex3f, glViewport
    from OpenGL.GL import GL_BLEND, GL_COLOR_BUFFER_BIT, GL_DEPTH_TEST, GL_FALSE                # pylint: disable=E0611
    from OpenGL.GL import GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_LUMINANCE, GL_LUMINANCE_ALPHA  # pylint: disable=E0611
    from OpenGL.GL import GL_MODELVIEW, GL_NO_ERROR, GL_ONE_MINUS_SRC_ALPHA                     # pylint: disable=E0611
    from OpenGL.GL import GL_PROJECTION, GL_RGB, GL_RGBA, GL_SRC_ALPHA, GL_QUADS                # pylint: disable=E0611
    from OpenGL.GL import GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER           # pylint: disable=E0611

    from OpenGL.GLU import gluBuild2DMipmaps, gluLookAt, gluPerspective

    from OpenGL.GLUT import GLUT_ALPHA, GLUT_DOUBLE, GLUT_RGBA, GL_UNSIGNED_BYTE
    from OpenGL.GLUT import GLUT_SCREEN_HEIGHT, GLUT_SCREEN_WIDTH, GLUT_WINDOW_X, GLUT_WINDOW_Y
    from OpenGL.GLUT import GLUT_WINDOW_WIDTH, GLUT_WINDOW_HEIGHT
    from OpenGL.GLUT import glutGetModifiers, GLUT_ACTIVE_ALT, GLUT_ACTIVE_CTRL, GLUT_KEY_F4
    from OpenGL.GLUT import glutCreateWindow, glutDisplayFunc, glutFullScreen, glutGet, glutIdleFunc
    from OpenGL.GLUT import glutInit, glutInitDisplayMode, glutInitWindowPosition, glutInitWindowSize
    from OpenGL.GLUT import glutKeyboardFunc, glutLeaveMainLoop, glutMainLoop, glutPositionWindow
    from OpenGL.GLUT import glutPostRedisplay, glutReshapeFunc, glutReshapeWindow, glutSwapBuffers
except ImportError, ex:
    raise ImportError("%s: %s\n\nPlease install PyOpenGL v3.0.1 or later: http://pyopengl.sourceforge.net\n" % (ex.__class__.__name__, ex))

getLogger().handlers[0].setFormatter(Formatter('%(asctime)s %(name)s %(levelname)s %(message)s', '%H:%M:%S')) # Restore conventional logging format after OpenGL has been loaded
logger = getLogger('signs')

LOAD_ALL = True # Load all textures to OpenGL, or only the ones currently used?

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

ESCAPE = '\x1b'
ENTER = '\x0d'
F4 = chr(GLUT_KEY_F4)

IMAGES_PATH = 'images/*.*'
SIGNS_PATH = '*.txt'

VERTICAL_FOV = 60

def sizeFormat(n, precision = 0):
    '''Returns a humanized string for a given amount of bytes.'''
    SUFFIXES = ('B', 'KB', 'MB', 'GB', 'TB','PB', 'EB', 'ZB', 'YB')
    if not n:
        return '0 B'
    logarithm = min(int(floor(log(n, 1024))), len(SUFFIXES) - 1)
    return '%.*f %s' % (precision, n / pow(1024, logarithm), SUFFIXES[logarithm])

def averageColor(color):
    '''Returns the average of the color components.'''
    return chr(int(round(float(sum(ord(c) for c in color)) / len(color))))

def addAlphaShader(color, alpha = '\xff'):
    '''If no alpha is present, adds it.'''
    return color if len(color) in (2, 4) else color + alpha if len(color) in (1, 3) else None

def grayScaleShader(color, alpha = '\xff'):
    '''Avarages color components to produce grayscale image with alpha.'''
    return color if len(color) == 2 else color + alpha if len(color) == 1 else averageColor(color) + alpha if len(color) == 3 else averageColor(color[:3]) + color[3] if len(color) == 4 else None

def transparentBlackShader(color):
    '''If no alpha is present, adds it, making pure black color transparent.'''
    return color if len(color) in (2, 4) else color + ('\xff' if any(ord(c) for c in color) else '\x00') if len(color) in (1, 3) else None

def maxAlphaShader(color):
    '''If no alpha is present, adds it, making it equal to the lightest component.'''
    return color if len(color) in (2, 4) else color + max(color) if len(color) in (1, 3) else None

def meanAlphaShader(color):
    '''If no alpha is present, adds it, making it equal to the average of components.'''
    return color if len(color) in (2, 4) else color + chr(int(round(float(sum(ord(c) for c in color)) / len(color)))) if len(color) in (1, 3) else None

class GLImage(object):
    NUM_TO_IMAGE_FORMATS = {1: GL_LUMINANCE, 2: GL_LUMINANCE_ALPHA, 3: GL_RGB, 4: GL_RGBA}
    GL_TO_IMAGE_FORMATS = {GL_LUMINANCE: 'L', GL_LUMINANCE_ALPHA: 'LA', GL_RGB: 'RGB', GL_RGBA: 'RGBA'}
    IMAGE_TO_GL_FORMATS = {'L': GL_LUMINANCE, 'LA': GL_LUMINANCE_ALPHA, 'RGB': GL_RGB, 'RGBA': GL_RGBA}

    @staticmethod
    def p2(x):
        '''Returns the first power of 2 greater or equal to x.'''
        i = 1
        while i < x:
            i *= 2
        return i

    @classmethod
    def getImage(cls, name, key = None, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def prepareImage(cls, w, h, data, padPixel = None, shader = None, *args, **kwargs):
        '''Processes image data with the specified shader and adds border
           of padPixel color (None means transparent black) around the image to make its size power of 2.'''
        (rw, rh) = tuple(cls.p2(d) for d in (w, h))
        left = (rw - w) // 2
        right = rw - w - left
        top = (rh - h) // 2
        bottom = rh - h - top
        pixelBytes = len(shader(data[0], *args, **kwargs) if shader else data[0])
        imageFormat = cls.NUM_TO_IMAGE_FORMATS[pixelBytes]
        if padPixel != None:
            assert len(padPixel) == pixelBytes
        else:
            padPixel = '\x00' * pixelBytes
        (lPad, rPad, tPad, bPad) = (p * padPixel for p in (left, right, top * rw, bottom * rw))
        lines = (chain.from_iterable((shader(d, *args, **kwargs) if shader else d) for d in data[i : i + w]) for i in xrange(0, w * h, w))
        data = ''.join(chain((tPad,), chain.from_iterable(chain((lPad,), line, (rPad,)) for line in lines), (bPad,)))
        return (rw, rh, imageFormat, data)

    def __init__(self, name, w, h, imageFormat, data):
        self.name = name
        self.w = w
        self.h = h
        self.imageFormat = imageFormat
        self.data = data
        self.dataSize = len(self.data)
        self.index = None

    def __str__(self):
        return '%s %dx%d %s (%s)' % (self.name, self.w, self.h, self.GL_TO_IMAGE_FORMATS[self.imageFormat], sizeFormat(self.getSize()))

    def getSize(self):
        return self.dataSize

    def loadTexture(self):
        if LOAD_ALL and self.index:
            return
        assert self.index == None
        self.index = glGenTextures(1)
        assert self.index > 0
        glBindTexture(GL_TEXTURE_2D, self.index)
        gluBuild2DMipmaps(GL_TEXTURE_2D, self.imageFormat, self.w, self.h, self.imageFormat, GL_UNSIGNED_BYTE, self.data)
        assert glGetError() == GL_NO_ERROR
        if LOAD_ALL:
            self.data = None

    def deleteTexture(self):
        if not LOAD_ALL:
            glDeleteTextures([self.index])
            self.index = None

class FileImage(GLImage):
    @classmethod
    def getImage(cls, name, key = None, shader = None, *args, **kwargs):
        try:
            image = imageOpen(name)
            data = tuple(''.join(chr(c) for c in d) for d in image.getdata())
            (w, h) = image.size
            (w, h, imageFormat, data) = cls.prepareImage(w, h, data, shader = shader)
            return cls(name, w, h, imageFormat, data)
        except:
            return None

class SignImage(GLImage):
    MARGIN_WIDTHS = (0.2, 0.3, 0.1, 0.3)
    BORDER_WIDTH = 0.1

    allHeight = None

    @classmethod
    def getImage(cls, name, textHeight, *args, **kwargs):
        margins = tuple(max(1, int(round(m * textHeight))) for m in cls.MARGIN_WIDTHS)
        border = max(1, int(round(cls.BORDER_WIDTH * textHeight)))
        hMargins = margins[1] + margins[3]
        vMargins = margins[0] + margins[2]
        hPadding = hMargins + 2 * border
        vPadding = vMargins + 2 * border
        font = dropwhile(lambda font: not font, (SysFont(fontName, textHeight, bold = True) for fontName in ('Verdana', 'Arial', None,))).next()
        fontRender = font.render('f %s g' % name, True, WHITE, BLACK) # f and g are added to provide proper line height
        array = array2d(fontRender) # Removing extra margins and the letters we added at the ends
        (wb, we) = cls.reduceStringIndexesByPatterns(''.join(chr(any(column)) for column in array), '\x01\x00\x01')
        (hb, he) = cls.reduceStringIndexesByPatterns(''.join(chr(any(line)) for line in array.T), '\x01')
        (w, h) = (we - wb + 1, he - hb + 1)
        surface = Surface((w + hPadding, h + vPadding)) # pylint: disable=E1121
        surface.fill(WHITE)
        surface.fill(BLACK, Rect(border, border, w + hMargins, h + vMargins))
        surface.blit(fontRender, (border + margins[3], border + margins[0]), Rect(wb, hb, w, h))
        data = tuple(chr(p % 256) * 2 for p in chain(*pixels2d(surface).T)) # LA image format
        (w, h, imageFormat, data) = cls.prepareImage(w + hPadding, h + vPadding, data)
        if cls.allHeight: # for debugging purposes
            assert h == cls.allHeight
        else:
            cls.allHeight = h
        return cls(name, w, h, imageFormat, data)

    @staticmethod
    def reduceStringIndexesByPatterns(s, patterns):
        '''Looks for the specified patterns from both ends of the string and returns resulting indexes.'''
        l = len(s)
        (start, end) = (0, l)
        for pattern in patterns:
            (start, end) = ((s.find(pattern, start, end) + 1 or start + 1) - 1, s.rfind(pattern, start, end) + 1 or end)
        return (start, end)

class CachedImage(GLImage):
    CACHE_LOCATION = 'cache'

    @classmethod
    def getImage(cls, name, key, imageType, *args, **kwargs):
        fileName = cls.getCacheFileName(name, key)
        try:
            image = imageOpen(fileName)
            (w, h) = image.size
            return cls(name, w, h, cls.IMAGE_TO_GL_FORMATS[image.mode], image.tostring(), 'loaded')
        except IOError:
            pass
        image = imageType.getImage(name, key, *args, **kwargs)
        if not image:
            return None
        mode = cls.GL_TO_IMAGE_FORMATS[image.imageFormat]
        fileImage = imageFromBuffer(mode, (image.w, image.h), image.data, 'raw', mode, 0, 1)
        dirName = dirname(fileName)
        if not isdir(dirName):
            makedirs(dirName)
        fileImage.save(fileName)
        return cls(image.name, image.w, image.h, image.imageFormat, image.data, 'saved')

    @classmethod
    def getCacheFileName(cls, name, key):
        return join(cls.CACHE_LOCATION, '%s_%s.png' % (name, key))

    def __init__(self, name, w, h, imageFormat, data, status):
        GLImage.__init__(self, name, w, h, imageFormat, data)
        self.status = status

    def __str__(self):
        return '(%s) %s' % (self.status, GLImage.__str__(self))

class ImageSource(object):
    def __init__(self, *args, **kwargs):
        self.images = []
        self.logger = getLogger(self.__class__.__name__)
        self.args = args
        self.kwargs = kwargs
        for image in self.load():
            if not image:
                break
            self.images.append(image)
            self.logger.info('%d> %s' % (self.numberOfImages(), str(image)))
            if LOAD_ALL:
                image.loadTexture()
            collect()
        num = self.numberOfImages()
        if num:
            self.logger.info("OK %d images loaded (%s)" % (num, sizeFormat(self.getSize())))
        else:
            self.logger.warning("No images loaded")

    def load(self): # generator
        raise NotImplementedError()

    def getRandomImage(self):
        return choice(self.images) if self.images else None

    def numberOfImages(self):
        return len(self.images)

    def getSize(self):
        return sum(image.getSize() for image in self.images)

class FileImageSource(ImageSource):
    def __init__(self, fileNames, imageClass = FileImage, *args, **kwargs):
        self.fileNames = fileNames
        self.imageClass = imageClass
        ImageSource.__init__(self, *args, **kwargs)

    def load(self): # generator
        self.logger.info("Processing %d images" % len(self.fileNames))
        for fileName in self.fileNames:
            yield self.imageClass.getImage(fileName, *self.args, **self.kwargs)

class SignImageSource(ImageSource):
    def __init__(self, fileNames, imageClass = SignImage, *args, **kwargs):
        self.fileNames = fileNames
        self.imageClass = imageClass
        ImageSource.__init__(self, *args, **kwargs)

    def load(self): # generator
        lines = (line.strip() for line in FileInput(self.fileNames)) if self.fileNames else ()
        allWords = chain.from_iterable(line.split() for line in lines if not line.startswith('#'))
        words = sorted(set(word for word in allWords if word.isalpha() and word.islower()))
        self.logger.info("Processing %d words" % len(words))
        for word in words:
            yield self.imageClass.getImage(word, *self.args, **self.kwargs)

class MultiImageSource(ImageSource):
    def __init__(self, *imageSources):
        self.imageSources = tuple([imageSource, xrange(0)] for imageSource in imageSources)
        self.numImages = 0
        self.totalSize = 0
        ImageSource.__init__(self)

    def load(self): # not a generator
        num = size = 0
        for t in self.imageSources:
            imageSource = t[0] # [imageSource, numberRange]
            newNum = num + imageSource.numberOfImages()
            t[1] = xrange(num, newNum)
            num = newNum
            size += imageSource.getSize()
        self.numImages = num
        self.totalSize = size
        assert self.numberOfImages() == sum(imageSource.numberOfImages() for (imageSource, _numbers) in self.imageSources)
        assert self.getSize() == sum(imageSource.getSize() for (imageSource, _numbers) in self.imageSources)
        return (None,)

    def getRandomImage(self):
        if not self.numberOfImages():
            return None
        num = randint(0, self.numberOfImages() - 1)
        for (imageSource, numbers) in self.imageSources: # ToDo: Optimize by using binary search
            if num in numbers:
                return imageSource.getRandomImage()
        assert False

    def numberOfImages(self):
        return self.numImages

    def getSize(self):
        return self.totalSize

class Sign(object):
    SIGN_HEIGHT_LIMITS = (100, 100)
    CYCLE_SEC_LIMITS = (90, 180)
    DISTANCE_LIMITS = (400, 20000)
    COLOR_SPEED_LIMITS = (0, 1)

    images = {}

    def __init__(self, image, disperse = True):
        self.image = image
        direction = choice((1, -1))
        self.angle = direction * (uniform(0.5 * pi, 1.5 * pi) if disperse else 0.5 * pi)
        self.h = uniform(*self.SIGN_HEIGHT_LIMITS)
        self.w = self.h * image.w / image.h
        self.omega = direction * 2 * pi / uniform(*self.CYCLE_SEC_LIMITS)
        self.r = randint(*self.DISTANCE_LIMITS)
        self.z = self.r * sin(uniform(-0.6, 0.6) * radians(VERTICAL_FOV))
        self.color = self.randomColor()
        self.targetColor = self.randomColor()
        self.colorSpeed = uniform(*self.COLOR_SPEED_LIMITS)
        counter = self.images.get(self.image, 0)
        if not counter:
            image.loadTexture()
        self.images[self.image] = counter + 1
        self.move(0)

    def __str__(self):
        return 'Sign<%s, %s, %s>' % (self.r, self.z, self.angle)

    def destroy(self):
        counter = self.images[self.image]
        if counter == 1:
            self.image.deleteTexture()
            del self.images[self.image]
        else:
            self.images[self.image] = counter - 1

    def adjustColor(self, c, t, dt):
        offset = self.colorSpeed * dt
        if abs(c - t) <= offset:
            return t
        return c + offset if c < t else c - offset

    def move(self, dt):
        self.angle += self.omega * dt
        if abs(self.angle) > 1.5 * pi:
            return False
        self.x = -self.r * cos(self.angle)
        self.y = self.r * sin(self.angle)
        self.color = tuple(self.adjustColor(c, t, dt) for (c, t) in zip(self.color, self.targetColor))
        if self.color == self.targetColor:
            self.targetColor = self.randomColor()
        return True

    def draw(self):
        glBindTexture(GL_TEXTURE_2D, self.image.index)
        glBegin(GL_QUADS)
        glColor3f(*self.color)
        glTexCoord2f(1, 1)
        glVertex3f(self.x, self.y - self.w, self.z - self.h)
        glTexCoord2f(1, 0)
        glVertex3f(self.x, self.y - self.w, self.z + self.h)
        glTexCoord2f(0, 0)
        glVertex3f(self.x, self.y + self.w, self.z + self.h)
        glTexCoord2f(0, 1)
        glVertex3f(self.x, self.y + self.w, self.z - self.h)
        glEnd()
        assert glGetError() == GL_NO_ERROR

    @staticmethod
    def randomColor():
        return hsv_to_rgb(random(), 1, 1)

class Scene(object):
    NUM_SIGNS = 250

    def __init__(self, imageSource):
        self.imageSource = imageSource
        self.signs = []

    def prepare(self):
        self.addSigns()

    def addSign(self, disperse = True):
        image = self.imageSource.getRandomImage()
        if not image:
            return None
        newSign = Sign(image, disperse)
        pos = 0
        for (pos, sign) in enumerate(self.signs): # ToDo: Optimize by using binary search
            if sign.r < newSign.r:
                break
        else:
            pos += 1
        self.signs.insert(pos, newSign)
        return newSign

    def addSigns(self, disperse = True):
        for _ in xrange(len(self.signs), self.NUM_SIGNS):
            if not self.addSign(disperse):
                return
        collect()
        assert len(self.signs) == sum(Sign.images.itervalues()) == self.NUM_SIGNS

    def move(self, dt):
        for sign in tuple(sign for sign in self.signs if not sign.move(dt)):
            self.signs.remove(sign)
            sign.destroy()
        self.addSigns(dt == 0)

    def draw(self):
        for sign in self.signs:
            sign.draw()

class Engine(object):
    def __init__(self, *args):
        self.fullScreen = True
        self.pause = False
        glutInit(*args)
        (sw, sh) = (glutGet(GLUT_SCREEN_WIDTH) or 800, glutGet(GLUT_SCREEN_HEIGHT) or 600)
        glutInitWindowSize(min(sw * 2 // 3, sh * 32 // 27), min(sw * 3 // 8, sh * 2 // 3))
        glutInitWindowPosition(100, 100)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA)
        glutCreateWindow("Wall of Signs")
        glutDisplayFunc(self.draw)
        glutIdleFunc(glutPostRedisplay)
        glutReshapeFunc(self.resize)
        glutKeyboardFunc(self.keyPressed)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glEnable(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def prepare(self, scene):
        self.scene = scene
        self.scene.prepare()
        self.lastTime = time()

    def draw(self):
        now = time()
        self.scene.move(0 if self.pause else now - self.lastTime)
        glClear(GL_COLOR_BUFFER_BIT)
        self.scene.draw()
        glutSwapBuffers()
        self.lastTime = now

    @staticmethod
    def resize(w, h):
        if h == 0:
            h = 1
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(VERTICAL_FOV, float(w) / h, 10, 100000)
        gluLookAt( 0, 0, 0,      1, 0,  0,   0, 0, 1)
        #gluLookAt(0, 0, 10000,  0, 0, 10,   1, 0, 0) # Top down view
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def getWindow(self):
        self.windowSize = (glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT))
        self.windowPosition = (glutGet(GLUT_WINDOW_X), glutGet(GLUT_WINDOW_Y))

    def setWindow(self):
        if self.fullScreen:
            self.getWindow()
            glutFullScreen()
        else:
            glutReshapeWindow(*self.windowSize)
            glutPositionWindow(*self.windowPosition)

    def keyPressed(self, *args):
        key = args[0]
        modifiers = glutGetModifiers()
        if (key == ESCAPE and not modifiers or
            key == F4 and modifiers == GLUT_ACTIVE_ALT or
            key in ('c', 'q', 'x') and modifiers in (GLUT_ACTIVE_ALT, GLUT_ACTIVE_CTRL)):
            if glutLeaveMainLoop:
                glutLeaveMainLoop()
            else:
                exit()
        elif key == ENTER and modifiers == GLUT_ACTIVE_ALT:
            self.fullScreen = not self.fullScreen
            self.setWindow()
        elif key == ' ':
            self.pause = not self.pause

    def run(self):
        self.getWindow()
        self.setWindow()
        glutMainLoop()

def signalHandler(signalNumber, currentStackFrame): # pylint: disable=W0613
    '''Signal handler, used to terminate program by SIGTERM.'''
    ImageSource.terminate = True
    exit(1)

def main(*args):
    try:
        signal(SIGTERM, signalHandler)
        seed()
        if LOAD_ALL:
            logger.info("Initializing OpenGL...")
            engine = Engine(*args)
        logger.info("Loading textures from %s..." % IMAGES_PATH)
        fileImageSource = FileImageSource(glob(IMAGES_PATH), CachedImage, None, FileImage, addAlphaShader)
        logger.info("Generating textures from %s..." % SIGNS_PATH)
        pyGameInit()
        # Max text and texture heights: 2:8, 11:16, 21:32 [ 43:64, 87:128, 172:256, 346:512 ] 694:1024, 891max:2048
        signImageSource = SignImageSource(glob(SIGNS_PATH), CachedImage, 346, SignImage)
        scene = Scene(MultiImageSource(fileImageSource, signImageSource))
        if not LOAD_ALL:
            logger.info("Initializing OpenGL...")
            engine = Engine(*args)
        engine.prepare(scene)
        logger.info("Starting rendering...")
        engine.run()
        logger.error("Unexpected exit from glutMainLoop()")
        exit(-2)
    except SystemExit, e:
        logger.warning("Terminated")
        exit(e.code)
    except KeyboardInterrupt:
        logger.warning("Ctrl-C")
        exit(1)
    except BaseException:
        logger.exception("Unexpected exception")
        exit(-1)

if __name__ == '__main__':
    main(*argv)
