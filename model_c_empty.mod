set INDEX;
set NEXAMPLES;
param FEATURES{NEXAMPLES,INDEX};
param LABELS{NEXAMPLES};
var z{NEXAMPLES} >= 0;
var b_0;
minimize obj : sum{i in NEXAMPLES}z[i]/card(NEXAMPLES);
subject to con1{i in NEXAMPLES} : LABELS[i]*(b_0) >= 1 - z[i];





