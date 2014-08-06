/*
TUIO C++ Example - part of the reacTIVision project
http://reactivision.sourceforge.net/

Copyright (c) 2005-2009 Martin Kaltenbrunner <mkalten@iua.upf.edu>

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files
(the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

Any person wishing to distribute modifications to the Software is
requested to send the modifications to the original developer so that
they can be incorporated into the canonical version.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "TuioDump.h"

//#include "PointMap.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <ctype.h>
#include <stdio.h>
#include <map>

// tuio
#include "TuioServer.h"

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/core/opengl_interop.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/video.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace TUIO;


/*void TuioDump::addTuioObject(TuioObject *tobj) {
std::cout << "add obj " << tobj->getSymbolID() << " (" << tobj->getSessionID() << ") "<< tobj->getX() << " " << tobj->getY() << " " << tobj->getAngle() << std::endl;
}

void TuioDump::updateTuioObject(TuioObject *tobj) {
std::cout << "set obj " << tobj->getSymbolID() << " (" << tobj->getSessionID() << ") "<< tobj->getX() << " " << tobj->getY() << " " << tobj->getAngle() 
<< " " << tobj->getMotionSpeed() << " " << tobj->getRotationSpeed() << " " << tobj->getMotionAccel() << " " << tobj->getRotationAccel() << std::endl;
}

void TuioDump::removeTuioObject(TuioObject *tobj) {
std::cout << "del obj " << tobj->getSymbolID() << " (" << tobj->getSessionID() << ")" << std::endl;
}

void TuioDump::addTuioCursor(TuioCursor *tcur) {
std::cout << "add cur " << tcur->getCursorID() << " (" <<  tcur->getSessionID() << ") " << tcur->getX() << " " << tcur->getY() << std::endl;
}

void TuioDump::updateTuioCursor(TuioCursor *tcur) {
std::cout << "set cur " << tcur->getCursorID() << " (" <<  tcur->getSessionID() << ") " << tcur->getX() << " " << tcur->getY() 
<< " " << tcur->getMotionSpeed() << " " << tcur->getMotionAccel() << " " << std::endl;
}

void TuioDump::removeTuioCursor(TuioCursor *tcur) {
std::cout << "del cur " << tcur->getCursorID() << " (" <<  tcur->getSessionID() << ")" << std::endl;
}

void  TuioDump::refresh(TuioTime frameTime) {
//std::cout << "refresh " << frameTime.getTotalMilliseconds() << std::endl;
}*/

int* i;
double pi = 3.1415;
int element_shape = MORPH_RECT;

map< int, Point2f> finger;
map<char,int>::iterator it;

cv::gpu::GpuMat g_frame, disparity,disparity2;
cv::gpu::GpuMat g_frameCanny;



static void OpenClose(Mat src, Mat dst)
{
	/*int n = open_close_pos - max_iters;
	int an = n > 0 ? n : -n;*/

	Mat element = getStructuringElement(element_shape, Size(8, 8) );//, Point(10, 10) );
	//Mat element = getStructuringElement(element_shape, Size(an*2+1, an*2+1), Point(an, an) );
	/*if( n < 0 )
	morphologyEx(src, dst, CV_MOP_OPEN, element);
	else*/
	morphologyEx(src, dst, CV_MOP_CLOSE, element);
	imshow("Open/Close",dst);
}


int main(int argc, char* argv[])
{

	

	int idn=0;
	vector<int> idNum[2];
	//cv::gpu::getCudaEnabledDeviceCount();
	//cout<<getCudaEnabledDeviceCount()<<endl;
	cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());


	cv::Mat frame;
	cv::VideoCapture reader(1);
	cv::Mat Showgframe;
	cv::Mat Nframe;
	cv::Mat Preframe;

	cv::gpu::GpuMat g_frame, disparity,disparity2;
	cv::gpu::GpuMat g_frameCanny;
	cv::TickMeter tm;

	std::vector<double> gpu_times;

	Size ksize;
	ksize.width=5;ksize.height=5;
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;

	CvFont initFont;
	cvInitFont(&initFont,CV_FONT_HERSHEY_PLAIN,1.0f,1.0f,1.0f,1,8);
	//VideoCapture cap;
	TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);
	Size subPixWinSize(10,10), winSize(31,31);

	const int MAX_COUNT = 500;
	bool needToInit = false;
	bool nightMode = false;
	Mat gray, prevGray, image;
	Point2f point;
	bool addRemovePt = false;

	
	//Tuio Server
	TuioServer *tuio = new TuioServer();
	TuioTime time;

	while(true)
	{

		vector<Point2f>centerName;
		vector<Vec4i> hierarchy;
		vector<vector<Point> > contours;

		reader.read(frame);
		cvtColor(frame,frame, CV_RGB2GRAY);
		imshow("source",frame);
		OpenClose( frame,frame );

		//GPU processing
		g_frame.upload(frame);
		ksize.width=3;ksize.height=3;
		gpu::bilateralFilter(g_frame,g_frame,3,3,3);
		gpu::GaussianBlur(g_frame,g_frame,ksize,3,0);
		gpu::threshold(g_frame,g_frame,150,255,THRESH_TRUNC);

		//ksize.width=5;ksize.height=5;
		gpu::GaussianBlur(g_frame,g_frame,ksize,5,0);
		gpu::threshold(g_frame,g_frame,80,255,1);
		gpu::GaussianBlur(g_frame,g_frame,ksize,3,0);
		//gpu::Canny(g_frame,g_frameCanny,80,80,3);


		cv::Mat Showgframe;
		g_frame.download(Showgframe);
		imshow("GPU",Showgframe);

		//findContours
		findContours( Showgframe, contours, hierarchy,  CV_RETR_LIST,CV_CHAIN_APPROX_NONE );


		vector<vector<Point> > contours_poly( contours.size() );
		vector<Rect> boundRect( contours.size() );
		vector<Point2f>center( contours.size() );
		vector<float>radius( contours.size() );
		vector<Rect> rect( contours.size() );

		RotatedRect rRect;

		for( int i = 0; i < contours.size(); i++ )
		{	approxPolyDP( Mat(contours[i]), contours_poly[i], 0.1, true );
		//boundRect[i] = boundingRect( Mat(contours_poly[i]) );
		minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i]);
		}

		Mat drawing = Mat::zeros( Showgframe.size(), CV_8UC3 );

		for( int i = 0; i< contours.size(); i++ )

		{	//circle( drawing, center[i], (int)radius[i], Scalar(255,255,0), 2, 8, 0 );
			if(contourArea(contours[i])>30){

				float radius = sqrt(contourArea(contours[i])/pi);
				centerName.resize(center.size());

				for ( int j =0 ;j< centerName.size();j++)
				{//cout<<"i= "<<j<<centerName[j]<<endl;
				}

				if(norm(centerName[i] - center[i])>20)
				{
					centerName[i]=center[i];
					float radius = sqrt(contourArea(contours[i])/pi);
					for ( int m =0 ;m< centerName.size();m++)
					{//cout<<"!!!!"<<"i= "<<m<<centerName[m]<<endl;
					}
				}

				circle( drawing, centerName[i], 1, Scalar(255,255,255) ,2, 8, 0 );
				
			}
		}
	
	drawing.copyTo(image);
		//drawing.copyTo(gray);
		cvtColor(image, gray, CV_RGB2GRAY);
		vector <string> idnum;
		Mat flow;

		cout<<" centerName: "<<centerName<<endl;

		if(finger.empty()){
			cout<<"finger is empty! "<<endl;
			for(int i=0; i<centerName.size(); i++) {
				finger[idn] = centerName[i];
				idn++;
			}
		}

		if(!finger.empty()){
				
			if( finger.size() > centerName.size() ){
				cout<<">"<<endl;
				vector<int> usedFingerID;
			
				if(centerName.empty()){
					for(std::map<int,Point2f>::iterator it=finger.begin(); it!=finger.end(); ++it){cout<<" fingerID: "<<it->first<<endl;}
					cout<<" CenterName = 0; "<<endl;
					finger.clear();
					
				}

				if(!centerName.empty()){
					for( int i=0; i<centerName.size(); i++) {
						int idtem;
						int min = 10000;		
					
						 for (std::map<int,Point2f>::iterator it=finger.begin(); it!=finger.end(); ++it){
							 if(norm( it->second - centerName[i])<min )
											{
												min = norm( it->second - centerName[i]);
												idtem = it->first;
												cout<< " idtem "<<idtem<<endl;
											}//if

								}
						 finger[idtem] = centerName[i];
						 usedFingerID.push_back(idtem);
						 
					}
				
				

				

				std::map<int,Point2f>::iterator it=finger.begin();
				while ( it != finger.end() ){
					int flag = 0;
					
						for ( int j=0; j< usedFingerID.size(); j++ ){
							
								if(it->first == usedFingerID[j]) flag = 1;
								else flag = 0;
						}
					
					if( flag == 0 ){
						std::map<int,Point2f>::iterator toErase = it;
						++it;
							
							finger.erase(toErase);
							cout<<"erase"<<endl;
							
						}
					else {++it;}
				}
			}
				
			}

			if( finger.size() == centerName.size() ){
				cout<<"="<<endl;
				for( int i=0; i<centerName.size(); i++) {
					int idtem;
					int min = 10000;		
					 for (std::map<int,Point2f>::iterator it=finger.begin(); it!=finger.end(); ++it){
						 if(norm( it->second - centerName[i])<min )
										{
											min = norm( it->second - centerName[i]);
											idtem = it->first;
											
										}//if

							}
					 finger[idtem] = centerName[i];
				}


			}

			if ( centerName.size() > finger.size() ){
				cout<<"<"<<endl;
				vector<int> usedID;
				for(std::map<int,Point2f>::iterator it=finger.begin(); it!=finger.end(); ++it){
					
					int idtem;
					for( int i=0; i<centerName.size(); i++) {
						int min = 10000;		
							 if(norm( it->second - centerName[i])<min )
											{
												min = norm( it->second - centerName[i]);
												idtem = i;
											}//if
							 
					}
					it->second = centerName[idtem];
					usedID.push_back( idtem );
					}
					for( int i=0; i<centerName.size(); i++) {
						int flag;
						for ( int j=0; j< usedID.size(); j++ ){
							if(i == usedID[j]) flag = 1;
							else flag = 0;
						}
						if( flag == 0 ){
							finger[idn] = centerName[i];
							idn++;
						}
					}
						
					
			}

			
		}//finger!=empty
		
							
		if(!finger.empty()){
		for (std::map<int,Point2f>::iterator it=finger.begin(); it!=finger.end(); ++it){
			cout<<"it->first: "<<it->first<<"it->second: "<<it->second<<endl;
			}
	
			vector<Point2f> PrePoints;
			vector<Point2f> AftPoints;
		
			vector<uchar> status;
			vector<float> err;
			if(prevGray.empty())
				gray.copyTo(prevGray);

		
			for (std::map<int,Point2f>::iterator it=finger.begin(); it!=finger.end(); ++it){
				PrePoints.push_back(it->second);
			}

			calcOpticalFlowPyrLK(prevGray, gray, PrePoints, AftPoints, status, err, winSize,
				4, termcrit, OPTFLOW_LK_GET_MIN_EIGENVALS, 0.001);
			
			
			int i=0;
			for (std::map<int,Point2f>::iterator it=finger.begin(); it!=finger.end(); ++it){
				it->second = AftPoints[i];
				i++;
			}
			cout<<"PrePoints: "<<PrePoints<<"  AftPoints:"<< AftPoints<<endl;
			
			for (std::map<int,Point2f>::iterator it=finger.begin(); it!=finger.end(); ++it){
				stringstream s;
				s<<"id="<<it->first;
				string ss;
				ss=s.str();
				putText(image, ss,it->second,FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,255,0),1,8,false);
				circle( image,it->second, 3, Scalar(100,100,255), -1, 8);
				
				
				time = TuioTime::getSessionTime();
				
				
				tuio -> initFrame(time);
				
			//	TuioCursor *n  = tuio -> addTuioCursor(it->second.x,it->second.y);
				//tuio->addCursorMessage(n);
				//TuioCursor::TuioCursor(it->first,it->first,it->second.x,it->second.y);
				TuioCursor *n  = new TuioCursor(it->first,it->first,it->second.x,it->second.y);
				//TuioCursor *i = tuio->addTuioCursor(2.3,3.3);
				tuio ->addExternalTuioCursor(n);
				//tuio->getTuioCursors();
				std::cout << "add cur " << n->getCursorID() << " (" << n->getSessionID() << ") " << n->getX() << " " << n->getY() << std::endl;
				//n->update(n);
				//n1.update ( 1.0,2.0, 3,4,5 );
				//std::cout << "set cur " << n->getCursorID() << " (" << n->getSessionID() << ") " << n->getX() << " " << n->getY() << std::endl;
				tuio->commitFrame();
				
				
			}

		//	cout<<"tuio->getTuioCursor(1)"<<tuio->getTuioCursor(1)<<endl;;


			for (std::map<int,Point2f>::iterator it=finger.begin(); it!=finger.end(); ++it){
			cout<<"it->first:!! "<<it->first<<"it->second:!! "<<it->second<<endl;
			}

			swap(prevGray, gray);
		}

		imshow("LK Demo", image);

		char c = (char)waitKey(10);
		if( c == 27 )
			break;
	}



	return 0;
}



/*if( argc >= 2 && strcmp( argv[1], "-h" ) == 0 ){
std::cout << "usage: TuioDump [port]\n";
return 0;

int port = 3333;
if( argc >= 2 ) port = atoi( argv[1] );

TuioDump dump;
TuioClient client(port);
client.addTuioListener(&dump);
client.connect(true);
}*/