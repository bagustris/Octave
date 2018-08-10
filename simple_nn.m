# simple_nn : program to solve nn
# input, X = [0 0 1; 1 0 1;  1 1 1; 0 1 1]
# output, y=[0; 1; 1; 0]

function[output] = simple(X, y, iter)

# initialize theta to random variales
theta1 = rand(3,3);
theta2 = rand(1,3);

# assign the transposed input to input layer (a{1})
a{1} = X';

# loop for carry out gradient descent
for i = 1:iter
	a{2} = 1 ./(1 + e.^-(theta1 * a{1}));
	a{3} = 1 ./(1 + e.^-(theta2 * a{2}));

	# backprop to calculate error
 	error3 = a{3} - y';
	error2 = (theta2' * error3) .* a{2} .*(1-a{2});

	# substract partical derivative from theta
	theta1 = theta1 - (error2 * a{1}');
	theta2 = theta2 - (error3 * a{2}');
end

output = a{3};
end
