# Explore data by
df.head() # gives you top5
df.info() # gives you rows and cols and type of the col, and hd
df.describe() # gives you count mean std min percentiles etc

# Visual exploration by:
plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

# Democrats vote for satellite
# Democrats vote for missile
# use plot.figure() to not have overlaying plots by reseting the graphics
