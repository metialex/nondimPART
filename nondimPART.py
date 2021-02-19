from tkinter import *
import numpy as np

#add new lines
#to check the link to the code

##Start
def calc():
    #initializing vars
    Env.d_p = float(diameter_inp.get())
    Env.mu = float(viscosity_inp.get())
    Env.rho_p = float(p_dens_inp.get())
    Env.rho_f = float(l_dens_inp.get())
    Env.g = float(gravity_inp.get())
    Env.d_p_n = float(diameter_n_inp.get())
    Env.g_n = float(gravity_n_inp.get())
    Env.omega = float(omega_inp.get())

    #define input array
    inputEnArray = [u_s_inp,Re_inp,viscosity_n_inp,Re_parties_inp, density_n_inp,u_s_n_inp]
    inputEnNondim = [Length_out, Time_out, Velocity_out, Pressure_out, Force_out, Torque_out]
    
    if(re_range_flag.get()==1 and stokes_iter_flag.get() == 0):
        Env.solve_stokes(inputEnArray,inputEnNondim)
    elif (re_range_flag.get()==1 and stokes_iter_flag.get() == 1):
        Env.solve_it(inputEnArray,inputEnNondim,1)
    elif(re_range_flag.get()==2):
        Env.solve_it(inputEnArray,inputEnNondim,2)


class environment():
    d_p = 0.0
    mu = 0.0
    rho_p = 0.0
    rho_f = 0.0
    g = 0.0
    u_s = 0.0

    d_p_n = 1.0
    mu_n = 0.0
    rho_s = 0.0
    g_n = 1.0
    u_s_n = 0.0
    Re = 0.0
    Re_parties = 0.0

    omega = 0.0 #porosity

    def solve_stokes(self,inputEnArray,inputEnNondim):
        self.rho_s=self.rho_p/self.rho_f
        self.u_s = ((self.rho_p-self.rho_f)*self.d_p*self.d_p*self.g)/(18*self.mu)
        self.Re = self.u_s*self.d_p*self.rho_f/self.mu
        self.mu_n= np.sqrt((self.rho_s-1)/(18*self.Re))
        self.Re_parties= 1/self.mu_n
        self.u_s_n=(self.rho_s-1)/(18*self.mu_n)

        self.setVar(inputEnArray[0],self.u_s)
        self.setVar(inputEnArray[1],self.Re)
        self.setVar(inputEnArray[2],self.mu_n)
        self.setVar(inputEnArray[3],self.Re_parties)
        self.setVar(inputEnArray[4],self.rho_s)
        self.setVar(inputEnArray[5],self.u_s_n)

        #inputEnNondim = [Length_out, Time_out, Velocity_out, Pressure_out, Force_out, Torque_out]
        self.setVar(inputEnNondim[0],self.d_p/self.d_p_n)
        self.setVar(inputEnNondim[1],(self.d_p*self.u_s_n)/(self.u_s*self.d_p_n)) #check it
        self.setVar(inputEnNondim[2],self.u_s/self.u_s_n)
        self.setVar(inputEnNondim[3],self.rho_f*pow(self.u_s,2)/pow(self.u_s_n,2))


    def solve_it(self,inputEnArray, inputEnNondim, formulaFlag):
        #_______________________________Compute u_s_______________________________
        def calcFh(u):
            Cd= (24*self.mu)/(u*self.rho_f*self.d_p)
            return -0.5*self.rho_f*u*u*Cd*(3.1415*0.25*pow(self.d_p,2))*(1-self.omega)
        def calcFh1(u):
            Re = (u*self.rho_f*self.d_p)/self.mu
            Cd= 24/Re*(1+0.1935*pow(Re,0.6305))
            return -0.5*self.rho_f*u*u*Cd*(3.1415*0.25*pow(self.d_p,2))*(1-self.omega)   
        #Cd-Re relation defined by Concha and Al-merdra (1979)
        def calcCd1(u,mu):
            Cd = 0.28*pow(1+(9.06/np.sqrt(u/mu)),2)
            return Cd
        #Cd-Re relation Clift, Grace and Weber (2005)
        def calcCd2(u,mu):
            Cd = 24/(u/mu)*(1+0.1935*pow(u/mu,0.6305))
            return Cd
        #Cd-Re relation for Stokes flow
        def calcCd3(u,mu):
            Cd = 24*mu/u
            return Cd

        self.rho_s=self.rho_p/self.rho_f

        m = ((3.1415*pow(self.d_p,3))/6)*(self.rho_p-self.rho_f)
        F_g = m*self.g
        u_new = 1e-15
        u_old = 1

        dt_1 = 1e-5

        while (abs(u_new-u_old) >1e-10):
            u_old = u_new
            if(formulaFlag == 1):
                Fh = calcFh(u_old)
            elif (formulaFlag == 2):
                Fh = calcFh1(u_old)
            u_new = (dt_1/m)*(F_g+Fh)+u_old
            #print(u_new)
        self.u_s = u_new
        self.Re = self.u_s*self.d_p*self.rho_f/self.mu

        #_______________________________Compute u_s_n and mu_n_______________________________
        #using Gauss-Siedel algorithm with damping coefficient
        u_new = 1
        u_old = 10

        mu_new = 1
        mu_old = 10

        dampCoeff = 0.9

        while (abs(u_new-u_old)*1e5 >1e-10):
            u_old = u_new
            mu_old = mu_new

            if (formulaFlag == 1):
                Cd = calcCd3(u_old,mu_old)
            elif (formulaFlag == 2):
                Cd = calcCd2(u_old,mu_old)

            u_new = u_old*(1-dampCoeff) + dampCoeff*np.sqrt((self.rho_s-1)*self.g_n/abs(0.75*Cd*(1-self.omega)))
            mu_new = mu_old*(1-dampCoeff) + dampCoeff*(self.d_p_n*self.mu*u_new)/(self.rho_f*self.u_s*self.d_p)
            
            #print(u_new, mu_new)
        
        self.mu_n = mu_new
        self.Re_parties =1/self.mu_n
        self.u_s_n = u_new

        #inputEnArray = [u_s_inp,Re_inp,viscosity_n_inp,Re_parties_inp, density_n_inp]
        self.setVar(inputEnArray[0],self.u_s)
        self.setVar(inputEnArray[1],self.Re)
        self.setVar(inputEnArray[2],self.mu_n)
        self.setVar(inputEnArray[3],self.Re_parties)
        self.setVar(inputEnArray[4],self.rho_s)
        self.setVar(inputEnArray[5],self.u_s_n)
        #inputEnNondim = [Length_out, Time_out, Velocity_out, Pressure_out, Force_out, Torque_out]
        self.setVar(inputEnNondim[0],self.d_p/self.d_p_n)
        self.setVar(inputEnNondim[1],(self.d_p*self.u_s_n)/(self.u_s*self.d_p_n)) #check it
        self.setVar(inputEnNondim[2],self.u_s/self.u_s_n)
        self.setVar(inputEnNondim[3],self.rho_f*pow(self.u_s,2)/pow(self.u_s_n,2))


    def setVar(self,inputEn,var):
        inputEn.configure(state="normal")
        inputEn.delete(0,END)
        inputEn.insert(0,str(round(var,8))) #change var
        inputEn.configure(state="readonly")


root =Tk()
root.title('nondimPART')
root.wm_attributes('-alpha', 0.7)
root.geometry('950x600')
root.resizable(width=False, height=False)

frame = Frame(root)
frame.place(relx=0.05,rely=0.05,relwidth=0.9,relheight=0.9)

Env = environment()

##______________________________Dimensional quantities______________________________##
label_0 = Label(frame,text='Dimensional quantities', font=('Verdana',12))
label_0.grid(row=0, sticky=W)

label_1 = Label(frame,text='Diameter [m]', font=('Verdana',10))
label_1.grid(row=1, sticky=W)

diameter_inp = Entry(frame,width = "10")
diameter_inp.grid(row=2, sticky=W)
diameter_inp.insert(END, '1e-5')

label_2 = Label(frame,text='Dynamic viscosity [N*s/m²]', font=('Verdana',10))
label_2.grid(row=3, sticky=W)

viscosity_inp = Entry(frame,width = "10")
viscosity_inp.grid(row=4, sticky=W)
viscosity_inp.insert(END,'1e-3')

label_3 = Label(frame,text='Particle density [kg/m³]', font=('Verdana',10))
label_3.grid(row=5, sticky=W)

p_dens_inp = Entry(frame,width = "10")
p_dens_inp.grid(row=6, sticky=W)
p_dens_inp.insert(END,'5e+3')

label_4 = Label(frame,text='Liquid density [kg/m³]', font=('Verdana',10))
label_4.grid(row=7, sticky=W)

l_dens_inp = Entry(frame,width = "10")
l_dens_inp.grid(row=8, sticky=W)
l_dens_inp.insert(END,'1e+3')

label_5 = Label(frame,text='Gravitational acceleration [m/s²]', font=('Verdana',10))
label_5.grid(row=9, sticky=W)

gravity_inp = Entry(frame,width = "10")
gravity_inp.grid(row=10, sticky=W)
gravity_inp.insert(END, '9.81')

label_11 = Label(frame,text='Settling velocity [m/s]', font=('Verdana',10))
label_11.grid(row=11, sticky=W)

u_s_inp = Entry(frame,width = "10")
u_s_inp.grid(row=12, sticky=W)
u_s_inp.configure(state="readonly")

omega_label = Label(frame,text='Omega (porousity)', font=('Verdana',10))
omega_label.grid(row=13, sticky=W)

omega_inp = Entry(frame,width = "10")
omega_inp.grid(row=14, sticky=W)
omega_inp.insert(END, '0.0')

##______________________________Non-Dimensional quantities______________________________##

label_6 = Label(frame,text='Non-Dim. quantities', font=('Verdana',12))
label_6.grid(row=0, column=2, sticky=W)

label_7 = Label(frame,text='Diameter', font=('Verdana',10))
label_7.grid(row=1, column=2, sticky=W)

diameter_n_inp = Entry(frame,width = "10")
diameter_n_inp.grid(row=2, column=2, sticky=W)
diameter_n_inp.insert(END,'1.0')
diameter_n_inp.configure(state="normal")

label_8 = Label(frame,text='Viscosity', font=('Verdana',10))
label_8.grid(row=3, column=2, sticky=W)

viscosity_n_inp = Entry(frame,width = "10")
viscosity_n_inp.grid(row=4, column=2, sticky=W)
viscosity_n_inp.configure(state="readonly")

label_9 = Label(frame,text='Density (rho_p/rho_f)', font=('Verdana',10))
label_9.grid(row=5, column=2, sticky=W)

density_n_inp = Entry(frame,width = "10")
density_n_inp.grid(row=6, column=2, sticky=W)
density_n_inp.configure(state="readonly")

label_10 = Label(frame,text='Gravity', font=('Verdana',10))
label_10.grid(row=7, column=2, sticky=W)

gravity_n_inp = Entry(frame,width = "10")
gravity_n_inp.grid(row=8, column=2, sticky=W)
gravity_n_inp.insert(END,'1.0')
gravity_n_inp.configure(state="normal")

label_12 = Label(frame,text='Settling velocity', font=('Verdana',10))
label_12.grid(row=9, column=2, sticky=W)

u_s_n_inp = Entry(frame,width = "10")
u_s_n_inp.grid(row=10, column=2, sticky=W)
u_s_n_inp.configure(state="readonly")

label_13 = Label(frame,text='Re number', font=('Verdana',10))
label_13.grid(row=11, column=2, sticky=W)

Re_inp = Entry(frame,width = "10")
Re_inp.grid(row=12, column=2, sticky=W)
Re_inp.configure(state="readonly")

label_14 = Label(frame,text='Re parties', font=('Verdana',10))
label_14.grid(row=13, column=2, sticky=W)

Re_parties_inp = Entry(frame,width = "10")
Re_parties_inp.grid(row=14, column=2, sticky=W)
Re_parties_inp.configure(state="readonly")


##______________________________Parties to real______________________________##

label_18 = Label(frame,text='Dim. factors      ', font=('Verdana',12))
label_18.grid(row=0, column=4, sticky=W)

label_19 = Label(frame,text='Length', font=('Verdana',10))
label_19.grid(row=1, column=4, sticky=W)

Length_out = Entry(frame,width = "10")
Length_out.grid(row=2, column=4, sticky=W)
Length_out.configure(state="readonly") 

label_20 = Label(frame,text='Time', font=('Verdana',10))
label_20.grid(row=3, column=4, sticky=W)

Time_out = Entry(frame,width = "10")
Time_out.grid(row=4, column=4, sticky=W)
Time_out.configure(state="readonly")

label_21 = Label(frame,text='Velocity', font=('Verdana',10))
label_21.grid(row=5, column=4, sticky=W)

Velocity_out = Entry(frame,width = "10")
Velocity_out.grid(row=6, column=4, sticky=W)
Velocity_out.configure(state="readonly")

label_23 = Label(frame,text='Pressure', font=('Verdana',10))
label_23.grid(row=7, column=4, sticky=W)

Pressure_out = Entry(frame,width = "10")
Pressure_out.grid(row=8, column=4, sticky=W)
Pressure_out.configure(state="readonly")

label_24 = Label(frame,text='Force', font=('Verdana',10))
label_24.grid(row=9, column=4, sticky=W)

Force_out = Entry(frame,width = "10")
Force_out.grid(row=10, column=4, sticky=W)
Force_out.configure(state="readonly")

label_22 = Label(frame,text='Torque', font=('Verdana',10))
label_22.grid(row=11, column=4, sticky=W)

Torque_out = Entry(frame,width = "10")
Torque_out.grid(row=12, column=4, sticky=W)
Torque_out.configure(state="readonly")

##______________________________Calculating options______________________________##
label_15 = Label(frame,text='Calculating options', font=('Verdana',12))
label_15.grid(row=0, column=5, sticky=W)

calculate= Button(frame,text='Calculate')
calculate.grid(row=1, column=5, sticky=W)
calculate.configure(command=calc)

#re_range = StringVar(root)
#re_range_option = OptionMenu(frame,re_range,"Re<1", "1<Re<1000")
#re_range_option.grid(row=2,column=4,sticky=W)

label_15 = Label(frame,text='Re number range', font=('Verdana',10))
label_15.grid(row=2, column=5, sticky=W)

re_range_flag = IntVar()
re_range_flag.set(1)
Radiobutton(frame, text="Re<1", variable=re_range_flag, value=1).grid(row=3, column=5, sticky=W)
Radiobutton(frame, text="1<Re<1000", variable=re_range_flag, value=2).grid(row=5, column=5, sticky=W)


cd_iter_flag    = IntVar()
cd_clift        = Checkbutton(frame,text="Clift et al.", variable = cd_iter_flag, onvalue = 1, offvalue = 0).grid(row = 6, column = 5)
cd_mordant      = Checkbutton(frame,text="Mordant et al.", variable = cd_iter_flag, onvalue = 1, offvalue = 0).grid(row = 7, column = 5)

#stokes_iter_flag = IntVar()
#stokes_iter = Checkbutton(frame,text="itter.", variable = stokes_iter_flag, onvalue = 1, offvalue = 0)
#stokes_iter.grid(row=3, column=6)

'''
label_17 = Label(frame,text='dt_1, dt_2', font=('Verdana',10))
label_17.grid(row=5, column=5, sticky=W)

dt_1_inp = Entry(frame,width = "5")
dt_1_inp.grid(row=6, column=5, sticky=W)
dt_1_inp.insert(END,'1e-5')

dt_2_inp = Entry(frame,width = "5")
dt_2_inp.grid(row=6, column=5)
dt_2_inp.insert(END,'1e-3')


label_16 = Label(frame,text='Settling velocity', font=('Verdana',10))
label_16.grid(row=7, column=4, sticky=W)


'''

##______________________________Other options______________________________##
col_count, row_count = frame.grid_size()
for row in range(row_count):
    frame.grid_rowconfigure(row,minsize=30)

for col in range(col_count):
    frame.grid_columnconfigure(col,minsize=50)



##______________________________Setup all variables______________________________##






##______________________________End of the loop______________________________##
root.mainloop()
