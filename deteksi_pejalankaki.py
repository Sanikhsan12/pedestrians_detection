# library
import cv2 as cv

# objek
cascade_path = (r"python-menengah\citra_digital\pedestrian\pedestrian.xml")
video_path = (r"python-menengah\citra_digital\pedestrian\Shopping, People, Commerce, Mall, Many, Crowd, Walking   Free Stock video footage   YouTube.mp4")
acuan_objek = cv.CascadeClassifier(cascade_path)
video =  cv.VideoCapture(video_path)

# function
def deteksi_orang(frame):
    grayscalling = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    orang = acuan_objek.detectMultiScale(grayscalling, scaleFactor=1.1)
    return orang

def drawer_box(frame):
    for x,y,w,h in deteksi_orang(frame):
        cv.rectangle(frame, (x,y) ,(x+w,y+h), (0,0,255), 5)

def close_window():
    video.release()
    cv.destroyAllWindows()
    exit()

def main():
    while True:
        _, frame = video.read()
        drawer_box(frame)
        cv.imshow("Deteksi Pedestrian",frame)

        # close window
        if cv.waitKey(1) & 0xFF == ord('q'):
            close_window()

# menjalakan programs
if __name__ == '__main__' :
    main()