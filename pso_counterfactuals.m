clc;
clear;
close all;

% Load the saved model
load('trainedModel.mat', 'net');


%Below are randmoly selected records and all have a default status
%Y=[23	1	1200	12	1	3	2	2	6	2	1	5]';
Y=[49	1	1445	18	1	4	3	1	3	3	2	5]';
%Y=[26	3	3181	24	1	2	2	2	3	2	2	1]';
%Y=[22	3	3973	14	1	1	4	2	0	3	3	1]';
%Y=[27	3	3416	27	1	3	3	2	3	3	2	1]';
%Y=[45	2	3049	18	1	2	2	2	2	2	2	1]';
%Y=[53	3	2424	24	1	5	2	4	3	3	2	5]';
%Y=[23	1	1498	12	1	3	3	2	3	2	2	1]';
%Y=[35	3	1592	12	1	4	2	4	2	2	2	4]';
%Y=[23	3	3919	36	1	3	1	2	3	3	2	1]';
%Y=[28	3	2284	24	1	4	3	2	3	3	2	1]';
%Y=[27	3	4576	45	1	1	3	4	1	3	2	2]';
%Y=[56	3	618	    12	1	5	1	4	3	3	2	1]';
%Y=[33	3	5150	24	1	5	3	4	2	3	2	1]';
%Y=[31	2	8947	36	1	4	3	3	1	3	2	5]';
%Y=[25	3	975 	15	1	3	2	4	2	1	2	1]';
%Y=[46	3	2611	24	1	5	1	4	3	4	2	1]';
%Y=[28	3	3595	36	1	5	3	2	3	3	2	1]';
%Y=[35	3	6948	36	1	3	3	2	1	3	1	1]';
%Y=[35	3	1471	15	1	3	4	4	3	3	3	1]';
%Y=[28	3	1323	6	1	5	3	4	0	1	2	2]';
%Y=[55	3	1424	12	1	5	1	2	2	2	2	5]';
%Y=[30	3	3566	48	1	4	3	1	9	3	2	2]';
%Y=[33	1	2186	15	1	4	1	2	2	2	1	5]';
%Y=[29	1	2333	24	1	2	2	3	2	3	2	5]';


%Please ensure you obtained these values (mad and md) using formulas from the paper
%mad_values = []


ub=[21,3,18424,72,2,5,4,4,10,2,3,5];
lb=[21,1,250 ,  4,1,1,1,0, 0,2,1,1];

positions=[1 10];

%replace immutable values to accomodate sampled data
T=Y';
for i=1:length(positions)
    ub(positions(i)) = T(positions(i));
    lb(positions(i)) = T(positions(i));
end

nVar=12;

%load nn_model

%model = nn_model;

model=net;


%Y is our starting point
rng default % For reproducibility


fobj = @(z) Optimizer(z,Y, mad_values, model);

% Define the PSO's paramters 
noP = 40;
maxIter = 50;
wMax = 0.9;
wMin = 0.2;
c1 = 2;
c2 = 2;
vMax = (ub - lb) .* 0.2; 
vMin  = -vMax;


% The PSO algorithm 

% Initialize the particles 
for k = 1 : noP
    Swarm.Particles(k).X = (ub-lb) .* rand(1,nVar) + lb; 
    Swarm.Particles(k).V = zeros(1, nVar); 
    Swarm.Particles(k).PBEST.X = zeros(1,nVar); 
    Swarm.Particles(k).PBEST.O = inf; 
    
    Swarm.GBEST.X = zeros(1,nVar);
    Swarm.GBEST.O = inf;
end


% Main loop
for t = 1 : maxIter
    
    % Calcualte the objective value
    for k = 1 : noP
        Swarm.Particles(k).X(1) = round( Swarm.Particles(k).X(1)  );
        Swarm.Particles(k).X(2) = round( Swarm.Particles(k).X(2)  );
        Swarm.Particles(k).X(3) = round( Swarm.Particles(k).X(3)  );
        Swarm.Particles(k).X(4) = round( Swarm.Particles(k).X(4)  );
        Swarm.Particles(k).X(5) = round( Swarm.Particles(k).X(5)  );
        Swarm.Particles(k).X(6) = round( Swarm.Particles(k).X(6)  );
        Swarm.Particles(k).X(7) = round( Swarm.Particles(k).X(7)  );
        Swarm.Particles(k).X(8) = round( Swarm.Particles(k).X(8)  );
        Swarm.Particles(k).X(9) = round( Swarm.Particles(k).X(9)  );
        Swarm.Particles(k).X(10) = round( Swarm.Particles(k).X(10)  );
        Swarm.Particles(k).X(11) = round( Swarm.Particles(k).X(11)  );
        Swarm.Particles(k).X(12) = round( Swarm.Particles(k).X(12)  );
        
        currentX = Swarm.Particles(k).X;
        
        Swarm.Particles(k).O = fobj(percDiff(currentX, Y')'); % Originally, the parameter passed was currentX
        
        % Update the PBEST
        if Swarm.Particles(k).O < Swarm.Particles(k).PBEST.O 
            Swarm.Particles(k).PBEST.X = percDiff(currentX, Y');  % Originally this value was currentX
            Swarm.Particles(k).PBEST.O = Swarm.Particles(k).O;
        end
        
        % Update the GBEST
        if Swarm.Particles(k).O < Swarm.GBEST.O
            Swarm.GBEST.X = percDiff(currentX, Y');  % Originally this value was currentX
            Swarm.GBEST.O = Swarm.Particles(k).O;
        end
    end
    
    % Update the X and V vectors 
    w = wMax - t .* ((wMax - wMin) / maxIter);
    
    for k = 1 : noP
        Swarm.Particles(k).V = w .* Swarm.Particles(k).V + c1 .* rand(1,nVar) .* (Swarm.Particles(k).PBEST.X - Swarm.Particles(k).X) ...
                                                                                     + c2 .* rand(1,nVar) .* (Swarm.GBEST.X - Swarm.Particles(k).X);
                                                                                 
        
        % Check velocities 
        index1 = find(Swarm.Particles(k).V > vMax);
        index2 = find(Swarm.Particles(k).V < vMin);
        
        Swarm.Particles(k).V(index1) = vMax(index1);
        Swarm.Particles(k).V(index2) = vMin(index2);
        
        Swarm.Particles(k).X = Swarm.Particles(k).X + Swarm.Particles(k).V;
        
        % Check positions 
        index1 = find(Swarm.Particles(k).X > ub);
        index2 = find(Swarm.Particles(k).X < lb);
        
        Swarm.Particles(k).X(index1) = ub(index1);
        Swarm.Particles(k).X(index2) = lb(index2);
        
    end
    
    outmsg = ['Iteration# ', num2str(t) , ' Swarm.GBEST.O = ' , num2str(Swarm.GBEST.O)];
    disp(outmsg);
    
    cgCurve(t) = Swarm.GBEST.O;
end
 
semilogy(cgCurve);
xlabel('Iteration#')
ylabel('Weight')

sparseness = Sparsity(Swarm.GBEST.X, Y');
similarity = Similarity(Swarm.GBEST.X, Y', mad_values);




