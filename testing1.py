from __future__ import division
import matplotlib
matplotlib.use('Agg')
import numpy
import scipy
from tsne import bh_sne
from matplotlib import pyplot as plt
#import matplotlib
#matplotlib.use('Agg')
import scipy.io as sio
import pickle
from sklearn.manifold import TSNE
import glob

cou=numpy.zeros((76,1))
a=glob.glob('filespickle/*.pickle')

#print(len(a))
path='/usr2/elizalde/NELS/users/rohan/fingerprinting/fp_NN_all/'
#print(len(a))
count12=0;
for i in range(len(a)):
        if i is 0:
#		print('o')
		fi=path+a[i]
#		print(fi)
		a23=pickle.load(open(fi,"rb"))
		b=numpy.array(a23)
		a_pram=b.shape[1]
		cou[count12]=a_pram
		#print(b.shape)
		count12=count12+1
		b00=numpy.reshape(b,(a_pram,128))
	else:
#		print('1')
#		print(a[i])
#		if a[i] == "filespickle/rooster_sound.pickle" :
#			print(i)
		fi1=path+a[i]
		a1=pickle.load(open(fi1,"rb"))
		b1=numpy.array(a1)
		b_pram=b1.shape[1]
#		print(count12)
		cou[count12]=cou[count12-1]+b_pram
		count12=count12+1
	#	if a[i]=="filespickle/rooster_sound.pickle":
	#		print(count12)
	#		print(i)
	#		print('good')
		b01=numpy.reshape(b1,(b_pram,128))
#		count12=count12+b_pram
		b00=numpy.concatenate((b00,b01),axis=0)

q,q23=pickle.load(open("intermediate_76_queries_new.pickle","rb"))
#print(q.shape)
q1=numpy.array(q)
q231=numpy.array(q23)
#print(q231)
sd_pram=q1.shape[0]
#print(sd_pram)
a98=numpy.zeros((sd_pram,1))
q22=numpy.reshape(q1,(sd_pram,128))
first_class_ap=numpy.zeros((1,1))
count_first_class_ap=numpy.zeros((1,1))
for jk in range(sd_pram):
	dist9=(cou-q231[jk])**2
#	dist99=numpy.sum(dist9,axis=1)
	dist999=numpy.sqrt(dist9)
	dist9999=numpy.argsort(numpy.transpose(dist999))
#	print(dist9999)
	a99=dist9999[0][0]
	if a99!=75:
		if (abs(cou[a99]-q231[jk])<=abs(cou[a99+1]-q231[jk]) and q231[jk]>cou[a99]):
			a99=a99+1
	a98[jk]=a99
	
	dist=(b00-q22[jk])**2
	qr=numpy.sum(dist,axis=1)
	rq=numpy.sqrt(qr)

	wer1=numpy.argsort(rq)
	wer=wer1[:100]
	a_pl=numpy.zeros((76,1))
	asr=numpy.zeros((1,1))
	for i in range(25):
		dist1=(cou-wer1[i])**2
 	#print(dist1.shape)
		dist11=numpy.array(dist1)
		dist2=numpy.sqrt(dist11)
	#print(dist2)
		e=numpy.argsort(numpy.transpose(dist2))
		a25=e[0][0]
		if a25!=75:
	#		print(e)
	#		print(cou[a25])
			if abs(cou[a25]-wer1[i])<= abs(cou[a25+1]-wer1[i]) and wer1[i]>cou[a25]:
				a25=a25+1
		
			a_pl[a25]=a_pl[a25]+1
	#	if a25==a98[jk]:
	#		count_total_ap=count_total_ap+1
		if a25==a98[jk]:
#			print(a98[jk])
#			print(a25)
			asr=asr+1
			first_class_ap=first_class_ap + (1/(i+1))
			count_first_class_ap=count_first_class_ap+1
			break
	if asr==0:
		count_first_class_ap=count_first_class_ap+1
		#else:
		#	count_first_class_ap=count_first_class_ap+1
print('first_instance_MAP')
print('count')
print(first_class_ap)
print(count_first_class_ap)
print(first_class_ap/count_first_class_ap)
#	a_pl_sor=numpy.argsort(numpy.transpose(a_pl))

a984=numpy.zeros((sd_pram,1))
#q22=numpy.reshape(q1,(sd_pram,128))
first_class_ap4=numpy.zeros((1,1))
count_first_class_ap4=numpy.zeros((1,1))


for jk in range(sd_pram):
        dist94=(cou-q231[jk])**2
#       dist99=numpy.sum(dist9,axis=1)
        dist9994=numpy.sqrt(dist94)
        dist99994=numpy.argsort(numpy.transpose(dist9994))
#       print(dist9999)
        a994=dist99994[0][0]
        if a994!=75:
                if (abs(cou[a994]-q231[jk])<=abs(cou[a994+1]-q231[jk]) and q231[jk]>cou[a994]):
                        a994=a994+1
        a984[jk]=a994

        dist4=(b00-q22[jk])**2
        qr4=numpy.sum(dist4,axis=1)
        rq4=numpy.sqrt(qr4)

        wer14=numpy.argsort(rq4)
        wer4=wer14[:100]
        a_pl4=numpy.zeros((76,1))
        asr4=numpy.zeros((1,1))
        for i in range(25):
                dist14=(cou-wer4[i])**2
        #print(dist1.shape)
                dist114=numpy.array(dist14)
                dist24=numpy.sqrt(dist114)
        #print(dist2)
                e4=numpy.argsort(numpy.transpose(dist24))
                a254=e4[0][0]
                if a254!=75:
        #               print(e)
        #               print(cou[a25])
                        if abs(cou[a254]-wer4[i])<= abs(cou[a254+1]-wer4[i]) and wer4[i]>cou[a254]:
                                a254=a254+1
#			a_pl4[a254]=a_pl4[a254]+1
        #       if a25==a98[jk]:
        #               count_total_ap=count_total_ap+1
                if a254==a984[jk]:
#                       print(a98[jk])
#                       print(a25)
                        asr4=asr4+1
                        first_class_ap4=first_class_ap4 + (1/(i+1))
                        count_first_class_ap4=count_first_class_ap4+1
#        		break
	if asr4==0:
                count_first_class_ap4=count_first_class_ap4+1
                #else:
                #       count_first_class_ap=count_first_class_ap+1
print('all_instance_MAP')
print('sum')
print(first_class_ap4)
print('count_first_class_ap4')
print(count_first_class_ap4)
print(first_class_ap4/count_first_class_ap4)


count_tot_ap=numpy.zeros((1,1))
count_total_class_ap=numpy.zeros((1,1))
countab=numpy.zeros((1,1))
z=25
z1=numpy.array(z)
a981=numpy.zeros((76,1))
for jk in range(sd_pram):
	dist91=(cou-q231[jk])**2
  #      dist99=numpy.sum(dist9,axis=1)
        dist9991=numpy.sqrt(dist91)
        dist99991=numpy.argsort(numpy.transpose(dist9991))
  #       print(dist9999)
        a991=dist99991[0][0]
        if a991!=75:
        	if (abs(cou[a991]-q231[jk])<=abs(cou[a991+1]-q231[jk]) and q231[jk]>cou[a991]):
                	a991=a991+1
         	a981[jk]=a991

	dist=(b00-q22[jk])**2
        qr=numpy.sum(dist,axis=1)
#108 #3print(qr.shape)
        rq=numpy.sqrt(qr)
#110 #print(rq.shape)
#111 #qs=numpy.min(rq)
#112 #print(wer[1])
        wer1=numpy.argsort(rq)
        wer=wer1[:100]
 #       print(wer)
	a_pl=numpy.zeros((76,1))
	count_total_class_ap=0
 	for i in range(25):
        	dist1=(cou-wer1[i])**2
         #print(dist1.shape)
                dist11=numpy.array(dist1)
                dist2=numpy.sqrt(dist11)
         #print(dist2)
                e=numpy.argsort(numpy.transpose(dist11))
                a25=e[0][0]
                if a25!=75:
         #               print(e)
         #               print(cou[a25])
                	if abs(cou[a25]-wer1[i])<= abs(cou[a25+1]-wer1[i]) and wer1[i]>cou[a25]:
                        	a25=a25+1

                        a_pl[a25]=a_pl[a25]+1
                if a25==a981[jk]:
                        # print(a98[jk])
                        # print(a25)
                       #  first_class_ap=first_class_ap + (1/(i+1))
                         count_total_class_ap=count_total_class_ap+1
                      #   break
#	print(count_total_class_ap)
	if count_total_class_ap!=0:
		#print(type(count_total_class_ap))
		count_tot_ap=count_tot_ap+(count_total_class_ap/25)
		countab=countab+1
print('total_class_MAP')
print('count')
print(count_tot_ap)
print(countab)
print(count_tot_ap/countab)
#print(count_tot_ap)
#print(countab)








'''
	coi=0
	sav=a_pl_sor[0][75]
	sav1=a_pl[sav]
	for ju in range(76):
		piu=a_pl_sor[0][75-ju]
		if sav1 !=a_pl[piu]:
			sav1=a_pl[piu]
			i0=numpy.where(a_pl[piu]==a_pl)		
#			print(i0)
		#	coi=coi+len(i0[0])
	#		if coi+len(i0[0])>25:
	#			rem=25-coi
	#			coi=coi+rem
	#			for er in range(rem):
	#				
	#				print(a[i0[0][er]])
#			else:		
			if a_pl[piu]>=1:

				print(a_pl[piu])
			for i in range(len(i0[0])):
				if (a_pl[piu]>=1):
					print(a[i0[0][i]])
		
'''
