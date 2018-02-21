# image-people-detection
Image detection in images using Convolutional Neural Network

## Usage instructions and comments:

### 1) Learning a model
Firstly, the learning dataset must be created in following structure:  
dataset/  
&nbsp;&nbsp;train/  
&nbsp;&nbsp;&nbsp;&nbsp;false/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;01.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;02.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  
&nbsp;&nbsp;&nbsp;&nbsp;true/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;01.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;02.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  
&nbsp;&nbsp;validation/  
&nbsp;&nbsp;&nbsp;&nbsp;false/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;01.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;02.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  
&nbsp;&nbsp;&nbsp;&nbsp;true/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;01.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;02.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  
&nbsp;&nbsp;test/  
&nbsp;&nbsp;&nbsp;&nbsp;false/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;01.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;02.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  
&nbsp;&nbsp;&nbsp;&nbsp;true/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;01.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;02.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  
Then set all the global variables in /src/training/training.py file and start learning.  

### 2) Model testing
In /src/training/test_set_prediction.py file set path to dataset/test directory and start

### 3) Prediction using the Sliding Window Method
Use /src/integration/sliding_window_integration.py file and set model path and destination path.

### 4) Prediction with the Selective Search Technique
Same as above. Use /src/integration/selective_search_integration.py file.

### 5) Detection in videos
Firstly, create directory with following structure:  
VIDEO/  
&nbsp;&nbsp;MP4/  
&nbsp;&nbsp;&nbsp;&nbsp;video.mp4 <!--- this is starting video --->  
&nbsp;&nbsp;IMG/ <!---  --->--ovdje se spremaju slike izrezane iz videoisječka
&nbsp;&nbsp;IMG_L/				--ovdje se spremaju slike sa oznakama detektiranih ljudi
&nbsp;&nbsp;MP4_L/				--ovdje se sprema videoisječak dobiven spajanjem slika iz mape iznad

-još je potrebno namjestiti puteve u video_integration programu
	-pokretanjem /src/integration/video_integration_sliding_window.py ili /src/integration/video_integration_selective_search.py dobivamo videoisječke s označenim ljudima 

================================================================================================================================================

U mapi /results nalaze se videoisječci i slike na kojima je provedena detekcija osoba.
