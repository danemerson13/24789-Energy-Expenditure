% The script further cleans up the imported table by the following:
% (1) Separates accelerometer by placement (Ankle, Chest, Thigh, Wrist)

function [] = chironTableFormat(file)

%% Import the table
path = strcat('Chiron','/',file,'.csv');
main = chironTableImport(path);

%% Remove Bodymedia rows to avoid time gap issue when filling missing activity labels
main(main.DeviceID == 'Bodymedia',:) = [];

%% Remove some unnecessary variables
main = removevars(main, {'VarName1','DeviceID','PersonID','Scenario','ZephyrBR','ZephyrST','ZephyrRR','BodyNBT','BodyST','BodyGSR','BodyCal','BodyMET','Android'});

%% Fill missing activity labels
main.Activity(main.Activity == '0') = categorical(NaN);
main.Activity = fillmissing(main.Activity, 'nearest');

%% Only consider the following activities {'walking','running','cycling'}
for act = {'walking','running','cycling'}
    % Create a new table for each activity
    tbl = main(main.Activity == act,:);

    %% Separate the accelerometer data by placement
    % Get the types of placements (0,Ankle, Chest, Thigh, Wrist)
%     imu_locs = categories(tbl.Placement);
    % Exclude 0, Chest, Wrist
%     imu_locs = imu_locs(2:end);
    imu_locs = {'ANKLE','THIGH'};
    
    for i = 1:numel(imu_locs)
        % Generate Boolean array for where Placement column matches a given
        % imu_loc
        idx = tbl.Placement == imu_locs{i};
        % Create new columns for AccXi for each imu_loc
        tbl = addvars(tbl, idx .* tbl.AccX, 'NewVariableNames', strcat(imu_locs{i}, '_AccX'));
        tbl = addvars(tbl, idx .* tbl.AccY, 'NewVariableNames', strcat(imu_locs{i}, '_AccY'));
        tbl = addvars(tbl, idx .* tbl.AccZ, 'NewVariableNames', strcat(imu_locs{i}, '_AccZ'));
    end
    
    % Now we can get rid of the original Placement and AccXi columns
    tbl = removevars(tbl, {'Activity','Placement','AccX','AccY','AccZ'});

    %% Convert times to seconds and zero based on the first time
    % First convert to seconds
    t = tbl.Time.Hour * 3600 + tbl.Time.Minute * 60 + tbl.Time.Second;
    % Now zero out according to the first time
    t = t - t(1);
    % Remove the old times from the table
    tbl = removevars(tbl, {'Time'});
    % Add the times to the table
    tbl = addvars(tbl, t, 'Before','ZephyrHR','NewVariableNames','Time');
    % Note - not simply writing over the old 'Time' column because this expects
    % a 'datetime' variable type but our new times have type 'double'

    %% Go through table and combine rows with the same time
    % Get the unique time stamps (C) and the row indices in the new compressed
    % table for all of the original rows (IC)
    [C,IA,IC] = unique(tbl{:,1},'rows');

    % Create a new array to hold the combined arrays
    combined = zeros(length(C),size(tbl,2));
    % Keep the unique times
    combined(:,1) = C;
    % Array with numerical data... computationally faster than querying
    % table with {i,j}
    num = tbl{:,2:end};
    
    % Loop through all the numerical rows and sum those w/ matching times
    for i = 1:length(IC)
        combined(IC(i),2:end) = combined(IC(i),2:end) + num(i,:);
    end
    
    % Take the mean of nonzero summed values by column
    for i = 1:size(combined,1)
        for j = 1:size(num,2)
            % Just a lazy check to replace i+1 with end
            if i == length(IA)
                nonzero = sum(num(IA(i):end,j) ~= 0);
            else
                nonzero = sum(num(IA(i):IA(i+1)-1,j) ~= 0);
            end
            % Take the mean of nonzero elements if there are any
            if nonzero > 0
                combined(i,j) = combined(i,j)/nonzero;
            end
        end
    end
    % Copy the old column names over
    varnames = tbl.Properties.VariableNames;
    % Concatenate the categorical and numerical data into one table
    tbl = array2table(combined);
    % Assign the old column names
    tbl.Properties.VariableNames = varnames;

    % Just clear out all our old variables
    clearvars -except tbl main varnames file act

    %% Now we want to interpolate the data so we have values in every cell
    % We have to be mindful of some big time gaps in the data - dont want to
    % interpolate across these
    gap = diff(tbl.Time);
    gap_idx = [1;find(gap > 1);size(tbl,1)];
    fprintf('Smallest gap: %g sec.\n', min(gap(gap > 1)))
    
    % Resampling frequency (in Hz) - Fs \in (0,inf)
    Fs = 2;
    
    % Initilize the new table
    new = table();
    
    for i = 1:length(gap_idx)-1
        if i == 1
            idxA = gap_idx(i);
            idxB = gap_idx(i+1);
        else
            idxA = gap_idx(i)+1;
            idxB = gap_idx(i+1);
        end
        % Check that the given interval is sufficiently long
        % Note: especially in the case of walking there are many short ~10
        % second intervals where the subject is walking around the
        % laboratory, we are only interested in the long interval walking
        % tests
        if tbl.Time(idxB) - tbl.Time(idxA)  < 60
            continue
        end
        % Store the old time vector for the given window
        t_old = tbl.Time(idxA:idxB);
        % Store all the numerical values to be interpolated for given window
        num = tbl{idxA:idxB,2:end};
        % Create a new vector of regularly spaced times
        n = floor((tbl.Time(idxB) - tbl.Time(idxA)) * Fs);
        t_new = linspace(tbl.Time(idxA),tbl.Time(idxB),n)';
        disp(['Time Elapsed: ', num2str(tbl.Time(idxB)-tbl.Time(idxA))])
        disp(['Length of Vector: ', num2str(length(t_new))])
        % Create a new array to hold the interpolated values
        interpolated = zeros(length(t_new),size(num,2));
    
        % Fill all zero values in num with NaN
        num(num == 0) = NaN;

        % Check that there are at least two COSMED and HR signals in the
        % interval, otherwise break and disregard this segment
        if sum(sum(~isnan(num)) <= 1) > 0
            disp('Not enough data to interpolate the given window')
            continue
        end
    
        for j = 1:size(num,2)
            val_idx = find(~isnan(num(:,j)) == 1);
            interpolated(:,j) = interp1(t_old(val_idx),num(val_idx,j),t_new,'linear');
            
            nan_idx = find(isnan(interpolated(:,j)));
            interpolated(nan_idx,:) = [];
            t_new(nan_idx,:) = [];
    
        end
        % Zero out the time vector based on the first time
        t_new = t_new - t_new(1);
    
        % Append the new times, and interpolated numerical
        % data altogether
        window = array2table(horzcat(t_new,interpolated));
        % Append the completed table for the given window to the overall
        % interpolated table
        new = [new; window];
    end
    new.Properties.VariableNames = varnames;
    if act == "walking"
        path = strcat('Aligned & Interpolated','/',num2str(Fs),'Hz','/',erase(file,'.csv'),'_Walk.csv');
        writetable(new,path);
    elseif act == "running"
        path = strcat('Aligned & Interpolated','/',num2str(Fs),'Hz','/',erase(file,'.csv'),'_Run.csv');
        writetable(new,path);
    elseif act == "cycling"
        path = strcat('Aligned & Interpolated','/',num2str(Fs),'Hz','/',erase(file,'.csv'),'_Cycle.csv');
        writetable(new,path);
    else
        error('Activity not valid')
    end
    clearvars -except main file
end
% alignedDataVisualization(file)
disp('All Done!')