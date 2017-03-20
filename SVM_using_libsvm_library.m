%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implement a 5-fold cross validation interface with the libsvm code 
%(do not use the one providedby the libsvm library). Train the SVM 
% with your code for values of ? : 0.25; 0:5; 1; 2; 4 and values
% of C:5; 1; 2; 4; 8; 16? Report the cross-validation error in a table
% with 30 entries. Pick the best combination of and C. 
% Train with all the data and report the training error and the test 
% error over the data in test.txt.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Start up
clear all
clc
disp('--- START ---')
%% Load data
train_data =dlmread('data_HW3\train.txt');
test_data =dlmread('data_HW3\test.txt');

%% Training using LibSVM 
train_fea = train_data(:,1:2);
training_label_vector = train_data(:,3);

L = length(train_data);
%% Perform 5 fold cross-validation
test_time = 50;
fold_num = 5;
% gamma = 0.25 0.5 1 2 4; c =0.5 1 2 4 8 15; 
gamma = [ 0.25 0.5 1 2 4];
C     = [ 0.5 1 2 4 8 15];
% radial basis function as kernal
min_acc = -inf;
for i = 1:length(gamma)
    for j = 1:length(C)
        for e = 1:test_time
            % generate k-fold index 
            k_idx = crossvalind('Kfold',L,fold_num);
            %  split training data to train and validation set
            for f = 1:fold_num
                train_fea_cv = train_fea(k_idx ~= f,:);
                train_label_cv = training_label_vector(k_idx ~= f);
                valid_fea_cv = train_fea(k_idx == f,:);
                valid_label_cv = training_label_vector(k_idx == f);
                options = ['-t 2 -g ',num2str(gamma(i)),' -c ',num2str(C(j))];
                model = svmtrain(train_label_cv,sparse(train_fea_cv),options);
                %% Test validation data to find out cross validation error          
               [pred_labels,accuracy,decision_values] = ...
                   svmpredict(valid_label_cv,sparse(valid_fea_cv), model);
                acc_f(f) = accuracy(1);
            end
            acc_e(e) = mean(acc_f);
        end
        acc(i,j) = mean(acc_e);
        if acc(i,j) > min_acc
            min_acc = acc(i,j);
            c_idx = j;
            g_idx = i;
        end
    end
end
disp('The best C and gamma combination is')
c_idx
g_idx

options = ['-t 2 -g ',num2str(g_idx),' -c ',num2str(c_idx)];
model = svmtrain(training_label_vector,sparse(train_fea),options);

test_fea = test_data(:,1:2);
test_label_vector = test_data(:,3);
[predicted_label, accuracy, decision_values] = ...
    svmpredict(test_label_vector,sparse(test_fea), model);