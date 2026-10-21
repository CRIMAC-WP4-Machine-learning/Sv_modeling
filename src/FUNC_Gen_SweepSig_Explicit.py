def func_Gen_SweepSig_Explicit(T_whole,T_center,T_effectiveDuration,Fs,F_start,F_end,Mexpo):
    import numpy as np
    import matplotlib.pyplot as plt
    #import math
    import scipy.fftpack
    from scipy import signal
    
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    T_whole=T_whole #  [s] total time signal   
    T_center=T_center # [s] the time where the center of pulse is located
    T=T_effectiveDuration # [s] Effective duration of the pulse
    Fs=Fs # [Hz] sampling frequency
    F1=F_start # [Hz] start of frequency sweep
    F2=F_end # [Hz] end of frequency sweep
    
    beta=np.pi*(F1+F2)
    alpha=np.pi*(F2-F1)/(T)
    dt=1/Fs
    mexpo=Mexpo # M exponent
    t=np.arange(-T_center,(T_whole-T_center),dt)
    t_vec=t+T_center
    y=np.exp(-(2*t/T)**(2*mexpo))*np.sin(alpha*t*t+beta*t)
    
#    FIG=plt.figure(figsize=(8,7)) 
#    ax = FIG.add_subplot(211)
#    plt.plot(t_vec,y)
#    
#    
#    DFT_y=scipy.fftpack.fft(y)
#    DF=1/(len(t)*dt)
#    Fvec=np.arange(0,Fs/2,DF)
#    ax = FIG.add_subplot(212)
#    plt.plot(Fvec,np.abs(DFT_y[0:len(Fvec)]))
    return t_vec,y
    
