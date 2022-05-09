% This function generates a usable table from the raw Chiron data which has
% some inconsistent data types per variable that are otherwise tricky to
% handle

function tbl = chironTableImport(file)

% Define table import options
% explicitly define data types so Matlab can handle them
opts = delimitedTextImportOptions("NumVariables", 21);
opts.DataLines     = [2, Inf];
opts.Delimiter     = ",";
opts.MissingRule = 'fill';
opts.VariableNames = ["VarName1", "DeviceID", "Time", "PersonID", "Scenario",...
    "Activity", "Placement", "AccX", "AccY", "AccZ", "ZephyrHR", "ZephyrBR",...
    "ZephyrST", "ZephyrRR", "BodyNBT", "BodyST", "BodyGSR", "BodyCal",...
    "BodyMET", "COSMED", "Android"];
opts.VariableTypes = ["double", "categorical", "datetime", "categorical",...
    "categorical", "categorical", "categorical", "double", "double",...
    "double", "double", "double", "double", "double", "double", "double",...
    "double", "double", "double", "double", "char"];

opts = setvaropts(opts, "Time", "InputFormat", "yyyy-MM-dd HH:mm:ss.SSS");

% import table
tbl = readtable(file, opts);

% Hardcode to remove sections of missing COSMED ground truth values that
% are found with the help of the below error flag. If left in, this causes
% errors with interpolation code. No point in keeping data that is missing
% ground truth labels anyways.
if file == "Chiron/PersonA.csv"
    tbl(966598:1051606,:) = [];
elseif file == "Chiron/PersonB.csv"
    tbl(1:203373,:) = [];
end

if sum(isnan(tbl.COSMED)) > 0
    error('Missing COSMED Values')
end