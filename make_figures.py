import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':13,'axes.spines.top':False,'axes.spines.right':False})
prices=[90,92,94,95,96,97,98,99,99.5,99.75,99.9]
wins=[p/(100-p) for p in prices]
fig,ax=plt.subplots(figsize=(9,5.2),dpi=150)
ax.plot(prices,wins,color='#1f4e79',lw=2.5,marker='o')
ax.set_yscale('log')
ax.set_xlabel('Price paid for the near certain side (US cents per share)')
ax.set_ylabel('Wins needed to cancel one loss (log scale)')
for p,w in [(95,19),(99,99),(99.5,199),(99.75,399),(99.9,999)]:
    ax.annotate(f'{w:,.0f}',(p,w),textcoords='offset points',xytext=(-6,9),ha='right',fontsize=12)
ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig('figures/breakeven_wins.png'); plt.close()
rows=[('1 day before scheduled end',-0.061,-0.291,0.127),
      ('7 days before scheduled end',-0.137,-0.421,0.093),
      ('30 days before scheduled end',0.166,-0.019,0.273),
      ('1 day before recorded close',0.068,-0.026,0.150),
      ('7 days before recorded close',-0.101,-0.302,0.074),
      ('30 days before recorded close',0.034,-0.185,0.207)]
fig,ax=plt.subplots(figsize=(9,5),dpi=150)
for i,(n,m,lo,hi) in enumerate(rows[::-1]):
    ax.plot([lo,hi],[i,i],color='#1f4e79',lw=3,solid_capstyle='round')
    ax.plot([m],[i],'o',color='#c0392b',ms=9)
ax.axvline(0,color='black',lw=1)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
ax.set_xlabel('Average gain per 100 US cents staked, in US cents')
ax.grid(axis='x',alpha=.25)
fig.tight_layout(); fig.savefig('figures/returns_by_timing.png'); plt.close()
print('ok')
