
param I; #number of scenarios
param J; #number of instruments
param tau;
param beta = (1 - tau)/tau; # parametr beta zależy od tau
param w_limit;

var w{1..J} >= 0;
var y; # zmienna y jest nieograniczona

var u{1..I} >= 0;
var v{1..I} >= 0;


param p{1..I};
param R{1..I, 1..J}; # rows as scenarios, columns as instruments



minimize EVaR_value:
	y;
	

subject to main_constraints{i in 1..I}: 
	p[i] * y - u[i] + v[i] >= -p[i] * sum{j in 1..J} (R[i,j] * w[j]);
	
subject to u_v_constraints: 
	sum{i in 1..I} u[i] - beta * sum{i in 1..I} v[i] >= 0;
	
subject to weights_sum:
	sum{j in 1..J} w[j] = 1;
	
subject to max_weight{j in 1..J}:
	w[j] <= w_limit;
	




