% clear;
% test_index = cell(5,1);
% train_index = cell(5,1);
% for ff = 1:1:5
%     test_index{ff} = (ff-1)*9*12+1:ff*9*12;
%     train_index{ff} = setdiff(1:5*9*12,test_index{ff});
% end
% save('crossvalididx_N_540_kfold_5.mat','train_index','test_index');



clear;
num_outer_loop = 9;
num_inner_loop = 2;
size_test = 5*12;   % 45/9 = 5
size_valid = 20*12;     % (45-5)/2 = 20
size_data = num_outer_loop*size_test;
data_index = 1:1:size_data;

test_index = cell(num_outer_loop,1);
train_index = cell(num_outer_loop,1);
nest_train_index = cell(num_outer_loop,num_inner_loop);
nest_valid_index = cell(num_outer_loop,num_inner_loop);

for ff_outer = 1:1:num_outer_loop

    test_index{ff_outer} = data_index((ff_outer-1)*size_test+1:ff_outer*size_test);
    train_index{ff_outer} = setdiff(data_index,test_index{ff_outer});
    train_data_index = train_index{ff_outer};
    for ff_inner = 1:1:num_inner_loop
        nest_valid_index{ff_outer,ff_inner} = train_data_index((ff_inner-1)*size_valid+1:ff_inner*size_valid);
        nest_train_index{ff_outer,ff_inner} = setdiff(train_data_index,nest_valid_index{ff_outer,ff_inner});
    end
end


save('nest_crossvalididx_N_540_kfold_9_vkfold_2.mat','train_index','test_index','nest_train_index','nest_valid_index');






