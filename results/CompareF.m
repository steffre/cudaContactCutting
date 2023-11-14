clear all;


baseFilename = 'forcesSeq_0';

currentFilename = [baseFilename, '.dat'];
    
f0{1} = load(currentFilename);

fpara = f0{1};


baseFilename = 'F';

currentFilename = [baseFilename, '.txt'];
    
f0{1} = load(currentFilename);

fseq = f0{1};


fdiff=fpara-fseq;




% comparison of the nodal position

baseFilename = 'updatep_0';

currentFilename = [baseFilename, '.dat'];
    
p0{1} = load(currentFilename);

p0_cpp = p0{1};


baseFilename = 'p';

currentFilename = [baseFilename, '.txt'];
    
p0{1} = load(currentFilename);

p0_MAT = p0{1};


pdiff=p0_cpp-p0_MAT;



% comparison of the spring initial lenght

baseFilename = 'initial_l0';

currentFilename = [baseFilename, '.dat'];
    
l0{1} = load(currentFilename);

l0_cpp = l0{1};


baseFilename = 'L0';

currentFilename = [baseFilename, '.txt'];
    
l0{1} = load(currentFilename);

l0_MAT = l0{1};


%l0diff=l0_cpp-l0_MAT;


% comparison p updated with cuda and Cpp

baseFilename = 'updatepCPP_0';

currentFilename = [baseFilename, '.dat'];
    
pcpp{1} = load(currentFilename);

p_cpp = pcpp{1};


baseFilename = 'updatepCuda_0';

currentFilename = [baseFilename, '.dat'];
    
pCuda{1} = load(currentFilename);

pCuda = pCuda{1};


pdiddcppcuda=p_cpp-pCuda;



