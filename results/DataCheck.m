baseFilename = 'initial_c';

currentFilename = [baseFilename, '.dat'];
    
c{1} = load(currentFilename);


baseFilename = 'initial_p0';

currentFilename = [baseFilename, '.dat'];
    
p0{1} = load(currentFilename);

p0 = p0{1};


% figure
% scatter3(p0(:,1),p0(:,2),p0(:,3))
% drawnow;
% axis equal;

baseFilename = 'initial_TopNodes';

currentFilename = [baseFilename, '.dat'];
    
TopNodes{1} = load(currentFilename);

TopNodes=TopNodes{1};

baseFilename = 'initial_BottomNodes';

currentFilename = [baseFilename, '.dat'];
    
BottomNodes{1} = load(currentFilename);

BottomNodes=BottomNodes{1};


p=p0;
p(:,4) = TopNodes;
p(:,5) = BottomNodes;


%%% -----------------------------------------------------------------------

baseFilename = 'initial_Nncon';

currentFilename = [baseFilename, '.dat'];
    
Nncon{1} = load(currentFilename);

Nnocn=Nncon{1};

baseFilename = 'initial_ncon';

currentFilename = [baseFilename, '.dat'];
    
ncon{1} = load(currentFilename);

nocn=ncon{1};

baseFilename = 'initial_l0';

currentFilename = [baseFilename, '.dat'];
    
l0{1} = load(currentFilename);

l0=l0{1};

%%% -----------------------------------------------------------------------

baseFilename = 'preIterAfterBC_1';

currentFilename = [baseFilename, '.dat'];
    
pd{1} = load(currentFilename);

pd = pd{1};


figure
scatter3(pd(:,1),pd(:,2),pd(:,3))
drawnow;
axis equal;


%%% -----------------------------------------------------------------------

baseFilename = 'file_1';

currentFilename = [baseFilename, '.dat'];
    
p1{1} = load(currentFilename);

p1 = p1{1};


figure
scatter3(p1(:,1),p1(:,2),p1(:,3))
drawnow;
axis equal;
