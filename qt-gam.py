t2,q2 = ida.Qt('pcon-gam2')
t4,q4 = ida.Qt('pcon-rgh1')
t8,q8 = ida.Qt('pcon-gam8')
t16,q16 = ida.Qt('pcon-gam16')
t40,q40 = ida.Qt('pcon-gam40')
t100,q100 = ida.Qt('pcon-gam100')
t200,q200 = ida.Qt('pcon-gam200')
t400,q400 = ida.Qt('pcon-gam400')
q2b = zeros(25)
q4b = zeros(25)
q8b = zeros(25)
q16b = zeros(25)
q40b = zeros(25)
q100b = zeros(25)
q200b = zeros(25)
q400b = zeros(25)
t4b = range(0,150,6)
t2b = t4b
t8b = t4b
t16b = t4b
t40b = t4b
t100b = t4b
t200b = t4b
t400b = t4b
for i in range(25):
    q2b[i] = q2[i*3]
    q4b[i] = q4[i*3]
    q8b[i] = q8[i*3]
    q16b[i] = q16[i*3]
    q40b[i] = q40[i*3]
    q100b[i] = q100[i*3]
    q200b[i] = q200[i*3]
    q400b[i] = q400[i*3]

#semilogy(t2,q2,color='000000')
semilogy(t4,q4,color='000000')
semilogy(t8,q8,color='000000')
semilogy(t16,q16,color='000000')
semilogy(t40,q40,color='000000')
#semilogy(t100,q100,color='000000')
#semilogy(t200,q200,color='000000')
semilogy(t400,q400,color='000000')
#semilogy(t2b,q2b,'*',markersize=12, color='000000',label=r'$\gamma=0.2l_p$')
semilogy(t4b,q4b,'o',markersize=12, color='000000',label=r'$\gamma=0.4l_p$')
semilogy(t8b,q8b,'s',markersize=12, color='000000',label=r'$\gamma=0.8l_p$')
semilogy(t16b,q16b,'d',markersize=12, color='000000',label=r'$\gamma=1.6l_p$')
semilogy(t40b,q40b,'^',markersize=12, color='000000',label=r'$\gamma=4l_p$')
#semilogy(t100b,q100b,'v',markersize=12, color='000000',label=r'$\gamma=10l_p$')
#semilogy(t200b,q200b,'<',markersize=12, color='000000',label=r'$\gamma=20l_p$')
semilogy(t400b,q400b,'>',markersize=12, color='000000',label=r'$\gamma=40l_p$')
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
ylabel(r'$\mathit{q/q_{0}}$',fontsize=32)
xlabel(r'$\mathit{t/t_{d}}$',fontsize=32)

gam = [0.2,0.4,0.8,1.6,4,10,20,40]
td_avg = np.array([94,90,94,101,104,92,90,100])
yerr = np.array([8.379,5.788,8.081,13.379,19.728,16.814,4.690,13.539])
#yerr_min = np.array([8,9,12,12,32,26,4,11])
#yerr_max = np.array([10,6,10,21,18,23,8,23])
fig = plt.figure()
ax = fig.add_subplot(111)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(4)

plt.show()
semilogx(gam,td_avg,'o',markersize=20,color='000000',mew=3)
plt.errorbar(gam,td_avg,yerr=yerr,fmt='o',color='000000',capsize=10, elinewidth=3)
ylim(65,125)
yticks((70, 80, 90, 100, 110, 120), (r'$70$', r'$80$', r'$90$', r'$100$', r'$110$', r'$120$'), fontsize=64)
#yticks((65, 75, 85, 95, 105, 115, 125), (r'$65$', r'$75$', r'$85$', r'$95$', r'$105$', r'$115$', r'$125$'), fontsize=64)
xticks((10**-1, 10**0, 10**1, 10**2), (r'$0.1$', r'$1$', r'$10$', r'$100$'), fontsize=64)
ax.xaxis.set_tick_params(length=32,width=4)
ax.yaxis.set_tick_params(length=32,width=4)
plt.show()


