from PIL import Image
#from PIL.Image import core as image
import os
import time


class BinAndCropClass():

    def __init__(self, path='../data/img/test/data28/pic.jpg'):
        self.path = path
    def checkPath(self,toPath="../data/img/train/collect/default/"):
        self.toPath = toPath
        self.state = os.path.exists(toPath)
        if self.state == False:
            os.makedirs(toPath)
            print(self.logtime() + " check the bin directory is exist:" + str(self.state) + " and mkdir path:" + toPath)
        else:
            print(self.logtime() + " check the bin directory is exist:" + str(self.state))

    def bin(self):
        # open the image
        img = Image.open(self.path)
        img = img.convert("L")
        img = img.resize((126, 126), Image.ANTIALIAS)
        #img = img.rotate(90)
        new_name = self.path[:24] + '2_2222'
        #print(new_name)
        #print(self.logtime()+" save file "+self.toPath+self.timename()+".jpg")
        #img.save(self.toPath+self.timename()+".jpg","JPEG")
        img.save(new_name, "JPEG")
        os.remove(self.path)

    def bincrop(self):
        # open the image
        img = Image.open(self.path)
        img = img.convert("L")
        img = img.resize((126, 126), Image.ANTIALIAS)
        #img = img.rotate(90)
        new_name = self.path[:24] + '2_2222'
        test_name= self.toPath+self.timename()
        #print(new_name)
        print(self.logtime() + " save file " + self.toPath + self.timename() + ".jpg")
        img.save(test_name+".jpg","JPEG")
        img.save(new_name, "JPEG")
        #img.save(test_name, "JPEG")
        os.remove(self.path)

    def timename(self):
        return str(time.time())

    def logtime(self):
        return time.strftime("%Y-%m-%d %H:%M %p", time.localtime())

if __name__ == '__main__':
    f = '../data/img/test/data28/pic.jpg'
    start = BinAndCropClass(f)
    start.bin()


