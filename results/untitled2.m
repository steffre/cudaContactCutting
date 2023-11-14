baseFilename = 'file_';
numIterations = 10;  % Update this if the number of files changes

% Create an empty cell array to hold all the matrices
allData = cell(1, numIterations);

for i = 1:numIterations-1
    % Construct filename for this iteration
    currentFilename = [baseFilename, num2str(i), '.dat'];
    
    % Read data from the file and store it in the cell array
    allData{i+1} = load(currentFilename);
    
    fprintf('Loaded data from %s\n', currentFilename);
end

% Now, you can access each matrix with allData{1}, allData{2}, etc.
sz = 25;
ColorPoint=ones(size(p,1),3).*[0,0.2,0.2];


p = allData{2};

scatter(p(:,1),p(:,2),p(:,3),sz,ColorPoint,'filled')