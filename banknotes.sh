#!/usr/bin/env bash

PY=$(cat <<EOF
---
title: "Banknote Classification"
output: html_document
---
<style>

  @font-face {
      font-family: 'Bebas Neue';
      src: url('Fonts/BebasNeue-Regular.ttf') format('truetype');
  }

  @font-face {
      font-family: "Arsenal";
      src: url('Fonts/Arsenal-Regular.ttf') format('truetype');
  }


  h1 {  font-family: "Bebas Neue"; }

  h3, h4, #names {  font-family: 'Arsenal'; }

  body, .r, .plot {
    background-color: #333;
    color: #ccc;
  }

  pre {
    background-color: #333 !important;
    color: #ddd !important;
  }

</style>
<h3>Differentiating between real and fake</h3>
<p>Sam Hopkins</p>

<h3>Initialize workspace and read data</h3>
\`\`\`{python}
#!/usr/bin/python
# Written by Sam Hopkins
import matplotlib.pyplot as plt
import numpy as np
import pandas as pnd
import math

data = pnd.read_csv("data_banknote_authentication.csv")
print(data.head())
\`\`\`

<h3>Calculate mean and SD of each attribute for real and counterfeit bills</h3>
\`\`\`{python}
real = data[data['counterfeit'] == 0]
counterfeit = data[data['counterfeit'] == 1]

uR = [real['variance'].mean(), real['skewness'].mean(), real['kurtosis'].mean(), real['entropy'].mean()]
uF = [counterfeit['variance'].mean(), counterfeit['skewness'].mean(), counterfeit['kurtosis'].mean(), counterfeit['entropy'].mean()]
sR = [real['variance'].std(), real['skewness'].std(), real['kurtosis'].std(), real['entropy'].std()]
sF = [counterfeit['variance'].std(), counterfeit['skewness'].std(), counterfeit['kurtosis'].std(), counterfeit['entropy'].std()]
\`\`\`

<h3>Predict whether each banknote is real or counterfeit using distance from means</h3>
\`\`\`{python}
predictions0 = []
for i, x in data.iterrows():
  distReal = np.sqrt((x['variance'] - uR[0])**2 + (x['skewness'] - uR[1])**2 + (x['kurtosis'] - uR[2])**2 + (x['entropy'] - uR[3])**2)
  distFake = np.sqrt((x['variance'] - uF[0])**2 + (x['skewness'] - uF[1])**2 + (x['kurtosis'] - uF[2])**2 + (x['entropy'] - uF[3])**2)
  predictions0.append(1 if distReal > distFake else 0)

colors0 = []
hits0 = []
for i, x in data.iterrows():
  if x['counterfeit'] == 0 and predictions0[i] == 0:
    hits0.append(1)
    colors0.append("#8888ff")
  elif x['counterfeit'] == 1 and predictions0[i] == 1:
    hits0.append(1)
    colors0.append("#ff8888")
  elif x['counterfeit'] == 1 and predictions0[i] == 0:
    hits0.append(0)
    colors0.append("#ffcccc")
  else:
    hits0.append(0)
    colors0.append("#ccccff")

accuracy0 = 100*sum(hits0)/len(hits0)
acc0Str = "{:.2f}".format(accuracy0)
\`\`\`

<h3>Plots</h3>
\`\`\`{python}
plt.scatter(data['variance'], data['skewness'], c=colors0)
plt.text(min(data['variance']) + 2, max(data['skewness']) + 3, "variance vs skewness accuracy " + acc0Str + "%", fontsize=12, verticalalignment='center')

plt.scatter(uR[0], uR[1], color="black", marker="x")
plt.text(uR[0]-1, uR[1]-1, 'Real', fontsize=12, verticalalignment='center')

plt.scatter(uF[0], uF[1], color="black", marker="x")
plt.text(uF[0]-1, uF[1]-1, 'Fake', fontsize=12, verticalalignment='center')

plt.show()


plt.scatter(data['variance'], data['kurtosis'], c=colors0)
plt.text(min(data['variance']) + 2, max(data['kurtosis']) + 3, "variance vs kurtosis accuracy " + acc0Str + "%", fontsize=12, verticalalignment='center')

plt.scatter(uR[0], uR[2], color="black", marker="x")
plt.text(uR[0]-1, uR[2]-1, 'Real', fontsize=12, verticalalignment='center')

plt.scatter(uF[0], uF[2], color="black", marker="x")
plt.text(uF[0]-1, uF[2]-1, 'Fake', fontsize=12, verticalalignment='center')

plt.show()


plt.scatter(data['variance'], data['entropy'], c=colors0)
plt.text(min(data['variance']) + 1, max(data['entropy']) + 1, "variance vs entropy accuracy " + acc0Str + "%", fontsize=12, verticalalignment='center')

plt.scatter(uR[0], uR[3], color="black", marker="x")
plt.text(uR[0]-1, uR[3]-1, 'Real', fontsize=12, verticalalignment='center')

plt.scatter(uF[0], uF[3], color="black", marker="x")
plt.text(uF[0]-1, uF[3]-1, 'Fake', fontsize=12, verticalalignment='center')

plt.show()


plt.scatter(data['skewness'], data['kurtosis'], c=colors0)
plt.text(min(data['skewness']) + 3, max(data['kurtosis']) + 2, "skewness vs kurtosis accuracy " + acc0Str + "%", fontsize=12, verticalalignment='center')

plt.scatter(uR[1], uR[2], color="black", marker="x")
plt.text(uR[1]-1, uR[2]-1, 'Real', fontsize=12, verticalalignment='center')

plt.scatter(uF[1], uF[2], color="black", marker="x")
plt.text(uF[1]-1, uF[2]-1, 'Fake', fontsize=12, verticalalignment='center')

plt.show()


plt.scatter(data['skewness'], data['entropy'], c=colors0)
plt.text(min(data['skewness']) + 2, max(data['entropy']) + 1, "skewness vs entropy accuracy " + acc0Str + "%", fontsize=12, verticalalignment='center')

plt.scatter(uR[1], uR[3], color="black", marker="x")
plt.text(uR[1]-1, uR[3]-1, 'Real', fontsize=12, verticalalignment='center')

plt.scatter(uF[1], uF[3], color="black", marker="x")
plt.text(uF[1]-1, uF[3]-1, 'Fake', fontsize=12, verticalalignment='center')

plt.show()


plt.scatter(data['kurtosis'], data['entropy'], c=colors0)
plt.text(min(data['kurtosis']) + 2, max(data['entropy']) + 1, "kurtosis vs entropy accuracy " + acc0Str + "%", fontsize=12, verticalalignment='center')

plt.scatter(uR[2], uR[3], color="black", marker="x")
plt.text(uR[2]-1, uR[3]-1, 'Real', fontsize=12, verticalalignment='center')

plt.scatter(uF[2], uF[3], color="black", marker="x")
plt.text(uF[2]-1, uF[3]-1, 'Fake', fontsize=12, verticalalignment='center')

plt.show()
\`\`\`

<h3>Predict whether each banknote is real or counterfeit using gaussian probability density classifier </h3>
\`\`\`{python}
predictions1 = []
for i, x in data.iterrows():
  ea1 = (-0.5) * ((x['variance'] - uR[0]) / sR[0])**2
  pRealV = (1/(sR[0] * math.sqrt(2*math.pi))) * math.exp(ea1)

  ea2 = (-0.5) * ((x['skewness'] - uR[1]) / sR[1])**2
  pRealS = (1/(sR[1] * math.sqrt(2*math.pi))) * math.exp(ea2)

  ea3 = (-0.5) * ((x['kurtosis'] - uR[2]) / sR[2])**2
  pRealK = (1/(sR[2] * math.sqrt(2*math.pi))) * math.exp(ea3)

  ea4 = (-0.5) * ((x['entropy'] - uR[3]) / sR[3])**2
  pRealE = (1/(sR[3] * math.sqrt(2*math.pi))) * math.exp(ea4)

  eb1 = (-0.5) * ((x['variance'] - uF[0]) / sF[0])**2
  pFakeV = (1/(sF[0] * math.sqrt(2*math.pi))) * math.exp(eb1)

  eb2 = (-0.5) * ((x['skewness'] - uF[1]) / sF[1])**2
  pFakeS = (1/(sF[1] * math.sqrt(2*math.pi))) * math.exp(eb2)

  eb3 = (-0.5) * ((x['kurtosis'] - uF[2]) / sF[2])**2
  pFakeK = (1/(sF[2] * math.sqrt(2*math.pi))) * math.exp(eb3)

  eb4 = (-0.5) * ((x['entropy'] - uF[3]) / sF[3])**2
  pFakeE= (1/(sF[3] * math.sqrt(2*math.pi))) * math.exp(eb4)

  pReal = pRealV * pRealS * pRealE * pRealK
  pFake = pFakeV * pFakeS * pFakeE * pFakeK
  predictions1.append(1 if pReal < pFake else 0)

colors1 = []
hits1 = []
for i, x in data.iterrows():
  if x['counterfeit'] == 0 and predictions1[i] == 0:
    hits1.append(1)
    colors1.append("#8888ff")
  elif x['counterfeit'] == 1 and predictions1[i]  == 1:
    hits1.append(1)
    colors1.append("#ff8888")
  elif x['counterfeit'] == 1 and predictions1[i]  == 0:
    hits1.append(0)
    colors1.append("#ffcccc")
  else:
    hits1.append(0)
    colors1.append("#ccccff")

accuracy1 = 100*sum(hits1)/len(hits1)
acc1Str = "{:.2f}".format(accuracy1)
\`\`\`

<h3>Plots</h3>
\`\`\`{python}
plt.scatter(data['variance'], data['skewness'], c=colors1)
plt.text(min(data['variance']) + 2, max(data['skewness']) + 3, "variance vs skewness accuracy " + acc1Str + "%", fontsize=12, verticalalignment='center')

plt.scatter(uR[0], uR[1], color="black", marker="x")
plt.text(uR[0]-1, uR[1]-1, 'Real', fontsize=12, verticalalignment='center')

plt.scatter(uF[0], uF[1], color="black", marker="x")
plt.text(uF[0]-1, uF[1]-1, 'Fake', fontsize=12, verticalalignment='center')

plt.show()


plt.scatter(data['variance'], data['kurtosis'], c=colors1)
plt.text(min(data['variance']) + 2, max(data['kurtosis']) + 3, "variance vs kurtosis accuracy " + acc1Str + "%", fontsize=12, verticalalignment='center')

plt.scatter(uR[0], uR[2], color="black", marker="x")
plt.text(uR[0]-1, uR[2]-1, 'Real', fontsize=12, verticalalignment='center')

plt.scatter(uF[0], uF[2], color="black", marker="x")
plt.text(uF[0]-1, uF[2]-1, 'Fake', fontsize=12, verticalalignment='center')

plt.show()


plt.scatter(data['variance'], data['entropy'], c=colors1)
plt.text(min(data['variance']) + 1, max(data['entropy']) + 1, "variance vs entropy accuracy " + acc1Str + "%", fontsize=12, verticalalignment='center')

plt.scatter(uR[0], uR[3], color="black", marker="x")
plt.text(uR[0]-1, uR[3]-1, 'Real', fontsize=12, verticalalignment='center')

plt.scatter(uF[0], uF[3], color="black", marker="x")
plt.text(uF[0]-1, uF[3]-1, 'Fake', fontsize=12, verticalalignment='center')

plt.show()


plt.scatter(data['skewness'], data['kurtosis'], c=colors1)
plt.text(min(data['skewness']) + 3, max(data['kurtosis']) + 2, "skewness vs kurtosis accuracy " + acc1Str + "%", fontsize=12, verticalalignment='center')

plt.scatter(uR[1], uR[2], color="black", marker="x")
plt.text(uR[1]-1, uR[2]-1, 'Real', fontsize=12, verticalalignment='center')

plt.scatter(uF[1], uF[2], color="black", marker="x")
plt.text(uF[1]-1, uF[2]-1, 'Fake', fontsize=12, verticalalignment='center')

plt.show()


plt.scatter(data['skewness'], data['entropy'], c=colors1)
plt.text(min(data['skewness']) + 2, max(data['entropy']) + 1, "skewness vs entropy accuracy " + acc1Str + "%", fontsize=12, verticalalignment='center')

plt.scatter(uR[1], uR[3], color="black", marker="x")
plt.text(uR[1]-1, uR[3]-1, 'Real', fontsize=12, verticalalignment='center')

plt.scatter(uF[1], uF[3], color="black", marker="x")
plt.text(uF[1]-1, uF[3]-1, 'Fake', fontsize=12, verticalalignment='center')

plt.show()


plt.scatter(data['kurtosis'], data['entropy'], c=colors1)
plt.text(min(data['kurtosis']) + 2, max(data['entropy']) + 1, "kurtosis vs entropy accuracy " + acc1Str + "%", fontsize=12, verticalalignment='center')

plt.scatter(uR[2], uR[3], color="black", marker="x")
plt.text(uR[2]-1, uR[3]-1, 'Real', fontsize=12, verticalalignment='center')

plt.scatter(uF[2], uF[3], color="black", marker="x")
plt.text(uF[2]-1, uF[3]-1, 'Fake', fontsize=12, verticalalignment='center')

plt.show()
\`\`\`

EOF
)

echo "$PY" > index.Rmd
Rscript -e "rmarkdown::render('index.Rmd')"
open index.html
