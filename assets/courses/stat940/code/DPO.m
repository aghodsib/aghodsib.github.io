% MATLAB Script to Compute the Given Expression

% Input values
pi_ref_w = 0.4; % π_ref(y_w|x)
pi_ref_l = 0.6; % π_ref(y_l|x)
pi_theta_w = 0.7; % π_theta(y_w|x)
pi_theta_l = 0.3    ; % π_theta(y_l|x)

% Temperature scaling factor
beta = 1; % You can change this value as needed

% Compute the logits for preferred and less-preferred responses
logit_w = beta * log(pi_theta_w / pi_ref_w); % log(π_theta(y_w|x) / π_ref(y_w|x))
logit_l = beta * log(pi_theta_l / pi_ref_l); % log(π_theta(y_l|x) / π_ref(y_l|x))

% Compute the difference between the logits
logit_diff = logit_w - logit_l;

% Compute the sigmoid of the logit difference
sigmoid_value = 1 / (1 + exp(-logit_diff))

% Compute the final value: log(σ(logit_diff))
result = -log(sigmoid_value)

% Display the result
%fprintf('The result is: %f\n', result);
