% This script is used for visualizing the original data from the JSI
% dataset

%% Load the person
personA = chironTableImport('Chiron/PersonA.csv');

%% Plot Accelerometer Data (AccX AccY AccZ)
figure(1)
subplot(3,1,1)
plot(personA{:,'Time'},personA{:,'AccX'})
title('AccX')
subplot(3,1,2)
plot(personA{:,'Time'},personA{:,'AccY'})
title('AccY')
subplot(3,1,3)
plot(personA{:,'Time'},personA{:,'AccZ'})
title('AccZ')

%% Plot Zephyr Data (HR, BR, ST, RR)
figure(2)
subplot(2,2,1)
plot(personA{:,'Time'},personA{:,'ZephyrHR'})
title('HR')
subplot(2,2,2)
plot(personA{:,'Time'},personA{:,'ZephyrBR'})
title('BR')
subplot(2,2,3)
plot(personA{:,'Time'},personA{:,'ZephyrST'})
title('ST')
subplot(2,2,4)
plot(personA{:,'Time'},personA{:,"ZephyrRR"})
title('RR')