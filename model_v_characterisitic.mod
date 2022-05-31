set INDEX;
set NEXAMPLES;
param FEATURES{NEXAMPLES,INDEX};
param LABELS{NEXAMPLES};
var b{INDEX};
var z{NEXAMPLES} >= 0;
var b_0;
minimize obj : sum{i in NEXAMPLES}z[i]/card(NEXAMPLES);
subject to con1{i in NEXAMPLES} : LABELS[i]*(sum{j in INDEX} b[j]*FEATURES[i,j] + b_0) >= 1 - z[i];





