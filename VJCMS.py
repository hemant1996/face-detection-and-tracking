# import the necessary packages
from face.facedetector import FaceDetector
import cv2
import numpy as np

#Initialize the paramters carefully as result depend on them

#Set the RATIO by which you want to resize the image. 
#Based on the video quality decide the 
RATIO = 1
#Enter the number of frames you want to track the face after detection
TRACK = 25
#Enter the number of frames to be skipped if no face is found
SKIP = 1
#Load the video. Enter the location of video on which you want to apply
cap = cv2.VideoCapture('videos/Video8.mp4')

"""def show_webcam(mirror=False):
    global cap;
    cap = cv2.VideoCapture(0)
    while True:
        ret_val, img = cap.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()"""
# initialize the termination criteria for CAMSHIFT, indicating
# a maximum of ten iteRATIOns or movement by a least one pixel
# along with the bounding box of the ROI
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

############################################################################################
#Apply Viola-Jones to detect faces in the video. The coordinates corresponding to the top left and bottom
#right pixel will be returned as list which will be used as an input in the subsequent frames to track.
def VJFindFace(frame):   
    #Include the global variables inside the scope of the function 
    global RATIO, orig
    #list to store the corner coordinates of the faces found.Initially empty
    allRoiPts = []    
    #generate a copy of the original frame
    orig = frame.copy()    
    #resize the original image. Set the aspect RATIO
    dim = (frame.shape[1]/RATIO, frame.shape[0]/RATIO);        
    # perform the actual resizing of the image and show it
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)                
    #convert the frame to gray scale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)        
    # find faces in the gray scale frame of the video using Haar feature based trained classifier
    fd = FaceDetector('cascades/haarcascade_frontalface_default.xml')
    faceRects = fd.detect(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (10, 10))    
    # loop over the faces and draw a rectangle around each
    for (x, y, w, h) in faceRects:
        #This step is extremely crucial. Here we are trying to decrease the size of the bounding box
        #for the face detected area. The primary reason for this being the box identified by VJ contains
        #a part of background. Hence when we find the mean of this box at the time of tracking, the resulting 
        #bounding box was much larger than the actual face region. Thus to eliminate the effect of background 
        #changing the mean we decrease the window size, as the color of interest will be available in this area
        #for tracking in the upcoming frames.
        x = RATIO*(x+20)
        y = RATIO*(y+20)
        w = RATIO*(w-25)
        h = RATIO*(h-25)            
        #Uncomment line 70, 76 and 77 to view the boxes around faces found using viola jones. Note that these
        #boxes will appear to be shifted and smaller size due to the opeRATIOn performed above
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        #Assign top left and bottom right pixel values each time Viola-Johnes is run
        #Append all the points detected for the face in the list
        allRoiPts.append((x, y, x+w, y+h))        
    #show the detected faces
    cv2.imshow("Faces", frame)
    cv2.waitKey(1)  
    return allRoiPts

####################################################################################################################
#Track the faces found using CAMSHIFT algorithm
def trackFace(allRoiPts, allRoiHist):        
    for k in range(0, TRACK):
        #read the frame and check if the frame is read. If these is some error reading the fram then return
        ret, frame = cap.read()
        if not ret:
            return -1;
            break
        i=0
        #convert the given frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #For histogram of each window found, back project them on the current frame and track using CAMSHIFT
        for roiHist in allRoiHist:            
            # Perform mean shift            
            backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
            # Apply cam shift to the back projection, convert the
            # points to a bounding box, and then draw them            
            #temp = allRoiPts[i]
            (r, allRoiPts[i]) = cv2.CamShift(backProj, allRoiPts[i], termination)  
            #Error handling for bound exceeding 
            for j in range(0,4):         
                if allRoiPts[i][j] < 0:
                    allRoiPts[i][j] = 0
            pts = np.int0(cv2.boxPoints(r))        
            #Draw bounding box around new position of the object
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
            i = i + 1            
        #show the face on the frame
        cv2.imshow("Faces", frame)
        cv2.waitKey(1)
    return 1;

def calHist(allRoiPts):
    global orig
    allRoiHist = []    
    #For each face found, convert it to HSV and calculate the histogram of
    #that region                                
    for roiPts in allRoiPts:                        
        # Grab the ROI for the bounding box by cropping and convert it
        # to the HSV color space.
        roi = orig[roiPts[1]:roiPts[-1], roiPts[0]:roiPts[2]]            
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)            

        # compute a HSV histogram for the ROI and store the
        # bounding box
        roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
        roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
        allRoiHist.append(roiHist);

    return allRoiHist
        
        
def justShow():
    global cap,SKIP
    #read the frame and display it for next SKIP number of frames
    for k in range(0,SKIP):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Faces", frame)
        cv2.waitKey(1)

# This function decides the flow of the overall algorithm. It contains the main decision making structure of the
# program. It is smart each to know which algorithm to run i.e. either Viola-Jones or CAMSHIFT or the skip the frame
# in the case of no face found.
def main():
    #Include the global varibles for manipulation
    global cap
    #show_webcam(mirror=True)
    i=0
    #While frames are present in the video
    while(cap.isOpened()):                
        #Try to find the faces using Viola-Jones. If faces are found, give the
        #pass to track it else for next five frames don't check any faces. Repeat until
        #a face is found in the frame
        if i % 2 == 0:
            #Before each call empty the pervious faces and their hsv histograms 
            allRoiPts = []
            allRoiHist = []
            
            #Read the frame and check if the frame is read. If these is some error reading the fram then return
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cv2.destroyAllWindows()
                return                
            #Capture the faces found in frame into a list                
            allRoiPts = VJFindFace(frame)
                                        
            #Check if faces are found in the given frame
            #If the face/faces are found 
            if len(allRoiPts) != 0:
                allRoiHist = calHist(allRoiPts)
                i=i+1
            else:
                #If no face is found display the next five frames without any processing 
                #To go for tracking in the next frame
                justShow()

        else:
            #Track the face found by viola jones for next TRACK number of frames using cam shift
            #print len(roiPts)
            error = trackFace(allRoiPts, allRoiHist)
            if error == -1:
                cap.release()
                cv2.destroyAllWindows()
                return
            i=i+1                

        #Exit on key press of q                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


# call main() function

if __name__ == "__main__":
    main()
