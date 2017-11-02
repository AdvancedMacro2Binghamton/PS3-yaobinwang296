%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ECON 634 Macro II
%%% Problem Set 3
%%% the Huggett Model
%%% Yaobin Wang
%%% 10/30/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Question 1
%%% Recursive Problem:
%%% v(s,a) = max_{a'? Gamma(s,a)} u[y(s)+a-q*a']+beta*E_{s'|s}[v(s',a')]
%%% where u(c) = c^(1-sigma)/(1-sigma), return function
%%%       y(e) = 1, y(u) = b
%%%       E_{s'|s}[v(s',a')] = sum_{s'} pi(s'|s)*v(s',a')
%%%
%%% State Variables:
%%% exogenous employment status s ?(e, u)
%%% current period asset a
%%%
%%% Control Variable:
%%% next period assst a'
%%%
%%% State Space:
%%% S*A
%%% S = {e, u}, A = [-2,5]
%%%
%%% Constraint Correspondence:
%%% Gamma(s,a)= [a_lo, (y(s)+a)/q]


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Question 2
clear all;
close all;
clc;

% PARAMETERS
beta = .9932; % discount factor 
sigma = 1.5; % coefficient of risk aversion
b = 0.5; % replacement ratio (unemployment benefits)
y_s = [1, b]; % endowment in employment states
PI = [.97 .03; .5 .5]; % transition matrix


% ASSET VECTOR
a_lo = -2; % lower bound of grid points
a_hi = 5; % upper bound of grid points
num_a = 1000;
a = linspace(a_lo, a_hi, num_a); % asset (row) vector

q_min = 0.98;
q_max = 1;

% ITERATE OVER ASSET PRICES
aggsav = 1 ;
tic;
while abs(aggsav) >= 0.01
    % INITIAL GUESS FOR q
    q_guess = (q_min + q_max) / 2;
    
    % CURRENT RETURN (UTILITY) FUNCTION
    cons = bsxfun(@plus,bsxfun(@minus,a',q_guess*a),permute(y_s, [1 3 2]));
    ret = (cons .^ (1-sigma)) ./ (1 - sigma); % current period utility
    ret (cons < 0) = -Inf; % negative consumption is impossible
    
    % INITIAL VALUE FUNCTION GUESS
    v_guess = zeros(2, num_a);
    
    % VALUE FUNCTION ITERATION
    dis1 = 1;
    tol = 1e-06;
    while dis1 > tol
        % CONSTRUCT RETURN + EXPECTED CONTINUATION VALUE (num_a*num_a*2)
        v = ret + beta * repmat(permute((PI*v_guess),[3 2 1]),[num_a 1 1]);
        
        % CHOOSE HIGHEST VALUE (ASSOCIATED WITH a' CHOICE)
        [vfn, p_indx] = max(v, [], 2); % size (vfn) = num_a 1 2
        
        % Distance between current guess and value function
        dis1 = max(max(abs(permute(vfn, [3 1 2])-v_guess)));
        
        % if dis1 > tol, update guess. O.w. exit.
        v_guess = permute(vfn, [3 1 2]);
    end
    
    % KEEP DECSISION RULE
    pol_indx = permute(p_indx,[3,1,2]);
    pol_fn = a(pol_indx); % size(pol_fn) = 2 num_a;
    
    % SET UP INITITAL DISTRIBUTION (uniform)
    Mu = ones(2, num_a)/(2*num_a);
    
    % ITERATE OVER DISTRIBUTIONS
    dis2 = 1;
    while dis2 >= tol
        [emp_ind, a_ind, mass] = find(Mu); % find non-zero indices
        MuNew = zeros(size(Mu));
        for ii = 1:length(emp_ind)
            % which a prime does the policy fn prescribe?
            apr_ind = pol_indx(emp_ind(ii), a_ind(ii)); 
            % which mass of households goes to which exogenous state?
            MuNew(:, apr_ind) = MuNew(:, apr_ind) + ... 
                (PI(emp_ind(ii), :) * mass(ii))';
        end
        dis2 = max(max(abs(MuNew-Mu)));
        Mu = MuNew;
    end
    
    % check for market clear
    aggsav = sum(sum(Mu.*pol_fn));
    
    % adjust q_guess
    if aggsav > 0
        q_min = q_guess;
    else
        q_max = q_guess;
    end 
end
time = toc;
q = q_guess;
fprintf('The steady state price is %1.4f.\n',q)

% plot wealth distribution
figure
plot(a+y_s(1,1),Mu(1,:),a+y_s(1,2),Mu(2,:),'r')
hold on
a1 = area(a+y_s(1,1),Mu(1,:),'FaceColor','b','EdgeColor','b');
a2 = area(a+y_s(1,2),Mu(2,:),'FaceColor','r','EdgeColor','r');
hold off
legend([a1 a2],'Employed','Unemployed')
xlabel('Wealth')
ylabel('share of the population')
title('Wealth Distribution')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Question 3
%%% Lorenz curve and Gini coeff. for wealth and earnings
d = reshape(Mu',[2*num_a 1]);
wealth = reshape(bsxfun(@plus, repmat(a, [2,1]), y_s')',[2*num_a 1]);
earnings = reshape(repmat(y_s',[1 num_a])',[2*num_a 1]);

d_wealth = cumsum(sortrows([d,d.*wealth,wealth],3));
L_wealth = bsxfun(@rdivide,d_wealth,d_wealth(end,:))*100;
Gini_wealth = 1-(sum((d_wealth(1:end-1,2)+d_wealth(2:end,2)).*...
    diff(d_wealth(:,1))))-(d_wealth(1,1)*d_wealth(1,2));

d_earnings = cumsum(sortrows([d,d.*earnings,earnings],3));
L_earnings = bsxfun(@rdivide,d_earnings,d_earnings(end,:))*100;
Gini_earnings = 1-(sum((d_earnings(1:end-1,2)+d_earnings(2:end,2)).*...
    diff(d_earnings(:,1))))-(d_earnings(1,1)*d_earnings(1,2));

fprintf('The Gini coefficient for wealth is %1.4f.\n',Gini_wealth)
fprintf('The Gini coefficient for earnings is %1.4f.\n',Gini_earnings)

refline = zeros(size(L_earnings(:,1),1)); % reference line, x=0

figure
plot(L_wealth(:,1),L_wealth(:,2),L_wealth(:,1),L_wealth(:,1),'--k',...
    L_wealth(:,1),refline,'k')
legend('Lorenz Curve','45 degree line','Location','NorthWest')
xlabel('Cumulative Share of Population')
ylabel('Cumulative Share of Wealth')
title('Lorenz Curve for Wealth')

figure
plot(L_earnings(:,1),L_earnings(:,2),L_earnings(:,1),L_earnings(:,1),...
    '--k',L_wealth(:,1),refline,'k')
legend('Lorenz Curve','45 degree line','Location','NorthWest')
xlabel('Cumulative Share of Population')
ylabel('Cumulative Share of Earnings')
title('Lorenz Curve for Earnings')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Question 4 Extra Credit
%%% If there was perfect insurance, all households would just consume the 
%%% average earnings weighted by the invariant transition matrix in 
%%% long-run. And the expected utility would be u(cons_bar), where cons_bar
%%% equals to the average earnings.

PI_invar = PI^100; % invariant transition matrix
cons_bar = PI_invar(1,1)*1+PI_invar(1,2)*b; % average earnings/consumption
W_FB = ((cons_bar^(1-sigma))/(1-sigma))/(1-beta);
lambda = (v_guess.^(-1)*W_FB).^(1/(1-sigma))-1;
b_pct = sum(sum((lambda>0).*Mu))*100;
wel_gain = sum(sum(lambda.*Mu));
fprintf('%2.1f percent of household would benefit from the plan.\n',b_pct)
fprintf('The economy-wide welfare gain is %1.4f.\n',wel_gain)

figure
plot(a,lambda(1,:),a,lambda(2,:))
hold on
vline(0,'--k','');
hline(0,'--k','');
hold off
legend('Employed','Unemployed','Location','NorthEast')
xlabel('Credit Level (a)')
ylabel('\lambda')
title('Consumption Equivalent')




