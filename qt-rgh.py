t001,q001 = ida.Qt('pcon-rgh001')
t005,q005 = ida.Qt('pcon-rgh005')
t01,q01 = ida.Qt('pcon-rgh01')
t02,q02 = ida.Qt('pcon-rgh02')
t05,q05 = ida.Qt('pcon-rgh05')
t1,q1 = ida.Qt('pcon-rgh1')
t2,q2 = ida.Qt('pcon-rgh2')
t5,q5 = ida.Qt('pcon-rgh5')
q001b = zeros(25)
q005b = zeros(25)
q01b = zeros(25)
q02b = zeros(25)
q05b = zeros(25)
q1b = zeros(25)
q2b = zeros(25)
q5b = zeros(25)
t001b = range(0,150,6)
t005b = t001b
t01b = t001b
t02b = t001b
t05b = t001b
t1b = t001b
t2b = t001b
t5b = t001b
for i in range(25):
    q001b[i] = q001[i*3]
    q005b[i] = q005[i*3]
    q01b[i] = q01[i*3]
    q02b[i] = q02[i*3]
    q05b[i] = q05[i*3]
    q1b[i] = q1[i*3]
    q2b[i] = q2[i*3]
    q5b[i] = q5[i*3]

semilogy(t001,q001,color='000000')
semilogy(t005,q005,color='000000')
#semilogy(t01,q01,color='000000')
semilogy(t02,q02,color='000000')
#semilogy(t05,q05,color='000000')
semilogy(t1,q1,color='000000')
#semilogy(t2,q2,color='000000')
semilogy(t5,q5,color='000000')
semilogy(t001b,q001b,'o',markersize=12,color='000000',label=r'$\sigma=0.001h_0$')
semilogy(t005b,q005b,'s',markersize=12,color='000000',label=r'$\sigma=0.005h_0$')
#semilogy(t01b,q01b,'s',markersize=12,color='000000',label=r'$\sigma=0.01h_0$')
semilogy(t02b,q02b,'d',markersize=12,color='000000',label=r'$\sigma=0.02h_0$')
#semilogy(t05b,q05b,'^',markersize=12,color='000000',label=r'$\sigma=0.05h_0$')
semilogy(t1b,q1b,'^',markersize=12,color='000000',label=r'$\sigma=0.1h_0$')
#semilogy(t2b,q2b,'<',markersize=12,color='000000',label=r'$\sigma=0.2h_0$')
semilogy(t5b,q5b,'>',markersize=12,color='000000',label=r'$\sigma=0.5h_0$')
legend(loc='upper left',prop={'size':24})
ax = plt.gca()
ax.tick_params(axis='both',reset=False,which='both',length=10,width=2,direction='bottom')
ax.yaxis.set_tick_params(length=24,width=2,direction='bottom')
ax.xaxis.set_tick_params(length=24,width=2,direction='bottom')
ax.tick_params(axis='both',reset=False,which='minor',length=10,width=1,direction='bottom')
ax.xaxis.set_tick_params(length=24,width=1,direction='bottom')
ax.yaxis.set_tick_params(length=24,width=1,direction='bottom')
yticks((10**0, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6), (r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$', r'$10^{5}$', r'$10^{6}$'), fontsize=32)
xticks((0, 50, 100, 150, 200), (r'$0$', r'$50$', r'$100$', r'$150$', r'$200$'), fontsize=32)
ylim(1,100)
xlim(0,150)
xlabel(r'$\mathit{t/t_{d}}$',fontsize=32)
ylabel(r'$\mathit{q/q_{0}}$',fontsize=32)

sig = [0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5]
td_avg = np.array([118,101,108,94,91,97,104,117])
yerr = np.array([7.530,4.604,9.555,3.847,6.535,5.0299,21.587,17.810])
#yerr_min = np.array([12,5,12,4,9,6,20,21])
#yerr_max = np.array([8,7,14,6,7,8,32,23])
fig = plt.figure()
ax = fig.add_subplot(111)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(4)

plt.show()
semilogx(sig,td_avg,'o',markersize=20,color='000000')
plt.errorbar(sig,td_avg,yerr=yerr,fmt='o',color='000000',capsize=10, elinewidth=3)
ylim(80,150)
xlim(0.0005,1)
yticks((85, 95, 105, 115, 125, 135, 145), (r'$85$', r'$95$', r'$105$', r'$115$', r'$125$', r'$135$', r'$145$'), fontsize=64)
#yticks((95, 105, 115, 125), (r'$95$', r'$105$', r'$115$', r'$125$'), fontsize=64)
xticks((10**-3, 10**-2, 10**-1, 10**0), (r'$0.001$', r'$0.01$', r'$0.1$', r'$1$'), fontsize=64)
ax.xaxis.set_tick_params(length=32,width=4)
ax.yaxis.set_tick_params(length=32,width=4)
plt.show()

