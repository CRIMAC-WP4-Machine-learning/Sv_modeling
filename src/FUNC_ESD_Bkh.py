def ESD_BKH(SigX,dt):
    import numpy as np
    import scipy.fftpack 
#%  this is a spectrum estimator for Transient signals
#%  Energy spectral density describes how the energy of a signal or a time series is distributed with frequency
#%% InPUTS
#% InpSig:  is the signal that its AutoPowerSpectrum is required.
#%          it should have size of  x  or 1 x N

#% dt:      (= 1/Fs) where Fs is sampling rate of data
#% WinYN:     determines whether Exponential window is used or not

#% Note: for frequency scaling: Frq=0:(1/(N*dt)):0.5/dt

# Function started

#import matplotlib.pyplot as plt


#%% Inputs to Check The Function
#t=1 # seconds
#Fs=1000 # sampling freq
#dt=1/Fs;
##N=t/dt
#F1=5
#F2=20.2

#%% Example to test the ESD function
#t=4 # time signal duratin "seconds"
#m=1 # Determines the rise time
#Fs=1000 # sampling freq
#dt=1/Fs;
#w0=24*(2*np.pi) # w0=f.2.pi
#n=8 # number of "n" denotes the approximate number of cycles at frequency w0
#timesig=np.arange(-t,t,dt)
#T=2*n*np.pi/w0 # Effective duration of the pulse
#
#timesig=np.arange(-t,t,dt)
#timesig=timesig.reshape(1,-1)
#
#E=np.exp(-( (2*timesig/T)**(2*m) ))
#plt.figure()
#plt.plot(timesig[0,:],E[0,:])
#plt.xlabel('Time (s)')
#plt.title('Envelope')
#plt.show() 
##timesig=np.linspace(0,t,N+1)
##SigX=0*np.sin(2*np.pi*F1*timesig)+3*np.sin(2*np.pi*F2*timesig)
##SigX=np.sin(w0*timesig)*np.exp(-( (w0*timesig/(n*np.pi))**(2*m) ))
#SigX=np.sin(w0*timesig)*E
#SUM=np.sum(dt*SigX*SigX)
#print('SUM: ',SUM)
#
#plt.figure()
#plt.plot(timesig[0,:],SigX[0,:])
#plt.xlabel('Time (s)')
#plt.title('Time Signal')
#plt.show() 
#%%
    SigX=SigX.reshape(1,-1)
    N=np.max(SigX.shape)
    
    Win=np.zeros(N)+1 
    Win=Win.reshape(1,-1)
    #   D = np.append(D,data[ii,:])[:,None] # stores in rows  (i.e. forms a column vector)   
    #   D = np.append(D,data[ii,:])[None,:] # stores in Columns  (i.e. forms a row vector)  
    WSig=Win*SigX
    WSig=WSig.reshape(1,-1)
    #plt.figure()
    #plt.plot(timesig[0,:],SigX[0,:])
    #plt.show() 
    Fs=1/dt
    DF=1/(N*dt)
    F_vec=np.arange(0,Fs/2,DF)
    F_vec=F_vec.reshape(1,-1)
    Nfvec=np.max(F_vec.shape)
    
    DFT_Wsig=scipy.fftpack.fft(WSig[0,:])  
    
    ESD=2*dt*dt*((np.absolute(DFT_Wsig[0:Nfvec]))**2)
    ESD[0]=0.5*ESD[0]
#ESD=ESD.reshape(1,-1)
#F_vec=F_vec.reshape(1,-1)
#plt.figure()
#plt.plot(F_vec[0,:],np.absolute(ESD[0,:]))
#plt.title('Energy Spectrum')
#plt.xlabel('Frequency (Hz)')
#plt.show() 
#
#AreaUnder=np.sum(ESD*DF)
#print('AreaUnder: ',AreaUnder)

#CUMSUM=np.cumsum(ESD*DF)
#CUMSUM=CUMSUM.reshape(1,-1)
#plt.figure()
#plt.plot(F_vec[0,:],CUMSUM[0,:])
#plt.title('Cumulative Energy')
#plt.xlabel('Frequency (Hz)')
#plt.show() 
    Output=np.column_stack((F_vec.reshape(-1,1),ESD.reshape(-1,1)))
    return Output
