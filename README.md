# Emotion Detection
 THIS　PROJECT IS KIND OF EMOTION RECOGNITION. IN THIS PROJECT, WE FOLLOW ONE-MODEL-FOR-ONE-USER PRINCIPLE,THAT' S, RECOGNITION IS NOT GENERAL-PURPOSED. THAT IS WHAT THIS MAKES THE MAJOR DIFFERENCE FROM OTHERS.
 
# Install

 Windows
 
    - [Install Anaconda](https://www.anaconda.com/download/)
    
          $ conda install python=3.5
          $ conda install -c menpo opencv3 
          $ conda install -c conda-forge dlib=19.4
          $ python -m pip install neurolab imutils
 Ubuntu
    
    - Install dlib, opencv, boost
    
          $ sudo apt-get install build-essential cmake libgtk-3-dev libboost-all-dev libopencv-dev
          $ wget https://bootstrap.pypa.io/get-pip.py
          $ sudo python get-pip.py
          $ sudo pip install numpy scipy scikit-image dlib
  
    - Install neurolab
     
          $ sudo pip install neurolab
    
 OSX

    - Install homebrew
    
          $ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
          $ brew update
          $ brew tap homebrew/science
          $ sudo nano ~/.bash_profile
          # Homebrew
          export PATH=/usr/local/bin:$PATH
          $ source ~/.bash_profile
 
    - [Install Anaconda & Downgrade python](https://www.continuum.io/downloads/)
      
          $ conda install python=3.5
            
    - Install opencv
      
          $ conda install numpy
          $ conda config --add channels conda-forge
          $ conda install opencv
          
    - Install dlib
      
          $ conda install dlib
            
    - Install neurolab, imutils
            
          $ sudo pip install neurolab imutils
        
# Usage
1. Download Landmark.dat
 
       $ ./download_data.sh
   
2. Train

        $python record.py
        
3. Test
        
        $python test.py
        
# License
 
  THE CODE IS NOT ALLOWED FOR COMMERCIAL PURPOSE. PLEASE CONTACT THE AUTHOR FIRST BEFORE YOU USE IT COMMERCIALLY.
 
# Thanks
