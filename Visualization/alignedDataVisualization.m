% This function is used for visualization of signals after preprocessing

function [] = alignedDataVisualization(file)

% file = 'PersonA';
i = 0;
for act = {'Walk','Run','Cycle'}
    path = strcat('Aligned & Interpolated','/','1Hz/',file,'_',act,'.csv');

    % Define table import options
    % explicitly define data types so Matlab can handle them
    opts = delimitedTextImportOptions("NumVariables", 9);
    opts.DataLines     = [2, Inf];
    opts.Delimiter     = ",";
    opts.MissingRule = 'fill';
    opts.VariableNames = ["Time","ZephyrHR","COSMED",...
        "ANKLE_AccX", "ANKLE_AccY", "ANKLE_AccZ",...
        "THIGH_AccX", "THIGH_AccY", "THIGH_AccZ"];
    opts.VariableTypes = ["double", "double","double",...
        "double", "double", "double",...
        "double", "double", "double"];
    
    tbl = readtable(path{1}, opts);

    lower = 0;
    upper = 50;
    
    figure(1+(i*3))
    plot(tbl.Time,tbl.ZephyrHR);
    xlabel('Time')
    xlim([lower,upper])
    title(strcat(act,' Zephyr HR'))
    
    figure(2+(i*3))
    plot(tbl.Time,tbl.COSMED)
    xlabel('Time')
    ylabel('MET')
    xlim([lower,upper])
    title(strcat(act,' MET'))
    
    figure(3+(i*3))
    title(strcat(act,' Accelerometer'))
    subplot(2,3,1)
    plot(tbl.Time,tbl.ANKLE_AccX)
    xlim([lower,upper])
    title('ANKLE_AccX','Interpreter','none')
    xlabel('Time')
    subplot(2,3,2)
    plot(tbl.Time,tbl.ANKLE_AccY)
    xlim([lower,upper])
    title('ANKLE_AccY','Interpreter','none')
    xlabel('Time')
    subplot(2,3,3)
    plot(tbl.Time,tbl.ANKLE_AccZ)
    xlim([lower,upper])
    title('ANKLE_AccZ','Interpreter','none')
    xlabel('Time')
    subplot(2,3,4)
    plot(tbl.Time,tbl.THIGH_AccX)
    xlim([lower,upper])
    title('THIGH_AccX','Interpreter','none')
    xlabel('Time')
    subplot(2,3,5)
    plot(tbl.Time,tbl.THIGH_AccY)
    xlim([lower,upper])
    title('THIGH_AccY','Interpreter','none')
    xlabel('Time')
    subplot(2,3,6)
    plot(tbl.Time,tbl.THIGH_AccZ)
    xlim([lower,upper])
    title('THIGH_AccZ','Interpreter','none')
    xlabel('Time')

    i = i + 1;
end