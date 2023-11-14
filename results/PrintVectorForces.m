clear all;

baseFilename = 'initial_p0';

currentFilename = [baseFilename, '.dat'];
    
p0{1} = load(currentFilename);

p0 = p0{1};




baseFilename = 'finalforce_1000';

currentFilename = [baseFilename, '.dat'];
    
f0{1} = load(currentFilename);

f0 = f0{1};


figure
scatter3(p0(:,1),p0(:,2),p0(:,3))
axis equal
hold on;




baseFilename = 'finalp_1000';

currentFilename = [baseFilename, '.dat'];
    
p1{1} = load(currentFilename);

p1 = p1{1};

scatter3(p1(:,1),p1(:,2),p1(:,3),"black")

quiver3(p1(:,1),p1(:,2),p1(:,3),f0(:,1),f0(:,2),f0(:,3),'r')