# -*- coding: utf-8 -*-

import os
import glob
import sys
import tables
import time
import re

import numpy as np
import nibabel as nib
import math
import subprocess
from PIL import Image



from keras.models import load_model
from dparams import dparams


import datetime
from keras import backend as K

modelfilepath = "Inceptioin_trans/inception_resnetv2_changed.h5";
niifilepath = [];



def arghash(args):
	ret = {};
	for ii in range(len(args)):
		if(re.search("^-",args[ii])):
			ret[args[ii]]  ="1";
			if(len(args) > ii+1):
				ret[args[ii]]  = args[ii+1];
	return ret;



def arrayTo255(array3d):
	hei = array3d.shape[0];
	wid = array3d.shape[1];
	dep = array3d.shape[2];
	valmax = array3d.max()
	valmin = array3d.min()
	
	offset = valmin;
	
	if(valmax-valmin == 0):
		sys.err.write("All elements in array have the same value.");
		return np.zeros(array3d.shape);
	
	mul = 255/(valmax-valmin);
	
	ret = np.add(array3d,-1*valmin);
	ret = np.multiply(ret,mul);
	return ret;
	

def arrayToImage(array2d_channels_last,path):
	hei = len(array2d_channels_last[0]);
	wid = len(array2d_channels_last);

	img = Image.new("RGB",(wid,hei))
	pixels = img.load()
	for xx in range(wid):
		for yy in range(hei):
			rdat = min(255,int(array2d_channels_last[xx,yy,0]+0.5));
			gdat = rdat;
			bdat = rdat;
			if(len(array2d_channels_last[xx,yy]) > 1):
				gdat = min(255,int(array2d_channels_last[xx,yy,1]+0.5));
			if(len(array2d_channels_last[xx,yy]) > 2):
				bdat = min(255,int(array2d_channels_last[xx,yy,2]+0.5));
			pixels[xx,yy] = (rdat,gdat,bdat);
	img.save(path);

model = load_model(modelfilepath)
#model.summary();


args = arghash(sys.argv);
filelists = [];

if "-niifile" in args:
	filelists.append(re.split(",",args["-niifile"]));

if "-list" in args:
	fin = open(args["-list"],"r");
	for ll in fin:
		filelists.append(re.split("[\s]+",re.sub("[\s]+$","",ll)));

fout = None;

if "-out" in args:
	fout = open(args["-out"],"w");

for niifilepath in filelists:
	ddata = [];
	try:
		for ff in niifilepath:
			img = nib.load(ff)
			ddata.append(arrayTo255(np.array(img.dataobj)));
			
		wid0 = ddata[0].shape[0];
		hei0 = ddata[0].shape[1];
		dep0 = ddata[0].shape[2];
		errorflag = "";
		for ii in range(len(ddata)):
			if(ddata[ii].shape[0] == wid0 and ddata[ii].shape[1] == hei0 and ddata[ii].shape[2] == dep0):
				continue;
			errorflag = niifilepath[ii];
		if(len(errorflag) > 0):
			sys.stderr.write(";".join(niifilepath));
			sys.stderr.write(errorflag+" image size is different.\n");
			continue;
		
	except:
		print("Unexpected error:", sys.exc_info()[0])
		continue;
	
	inputwidth = dparams["image_shape"][0];
	inputheight = dparams["image_shape"][1];
	
	
	offsetx = int((dparams["image_shape"][0] -ddata[0].shape[0])/2);
	offsety = int((dparams["image_shape"][1] -ddata[0].shape[1])/2);
	
	
	if(ddata[0].shape[0] >  dparams["image_shape"][0] or ddata[0].shape[1] >  dparams["image_shape"][1]):
		sys.stderr.write(niifilepath[0]+" The width or height of the image is larger than expected. Data out of range is discarded.\nwid: "+str(ddata[0].shape[0])+" vs "+str(dparams["image_shape"][0])+"\n");
		sys.stderr.write("hei: "+str(ddata[0].shape[1])+" vs "+str(dparams["image_shape"][1])+"\n");
	
	
	for zz in range(ddata[0].shape[2]):
		inarray = np.zeros(shape=(1,dparams["image_shape"][0],dparams["image_shape"][1],3));#モデルに入力するための配列
		for dx in range(ddata[0].shape[0]):
			xx = dx+offsetx;
			if(xx < 0 or xx >= inputwidth):
				continue;
			for dy in range(ddata[0].shape[1]):
				yy = dy+offsety;
				if(yy < 0 or yy >= inputheight):
					continue;
				
				dval = ddata[0][dx,dy,zz];
				inarray[0,xx,yy,0] = dval;
				inarray[0,xx,yy,1] = dval;
				inarray[0,xx,yy,2] = dval;
				
				rfile = niifilepath[0];
				gfile = niifilepath[0];
				bfile = niifilepath[0];
				
				
				if(len(ddata) > 1):
					inarray[0,xx,yy,1] = ddata[1][dx,dy,zz];
					gfile = niifilepath[1]
				if(len(ddata) > 2):
					inarray[0,xx,yy,2] = ddata[2][dx,dy,zz];
					bfile = niifilepath[2]
		
		pred =  model.predict(inarray);
		
		reslist = [];
		for dd in pred[0]:
			reslist.append(str(dd));
		
		usedfiles = [rfile,gfile,bfile];
		if(fout == None):
			print(",".join(usedfiles)+"\t"+str(zz),end="\t");
			print("\t".join(reslist));
			
		else:
			fout.write(",".join(usedfiles)+"\t"+str(zz)+"\t"+"\t".join(reslist)+"\n");
			
		

if(not fout == None):
	fout.close();

