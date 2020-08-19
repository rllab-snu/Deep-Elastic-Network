clc; clear all; close all;

ftr = fopen('./train_mini.txt');
Ctr = textscan(ftr, '%s %d %d');

fte = fopen('./val_mini.txt');
Cte = textscan(fte, '%s %d %d');

for i = 1:10
    mtr = fopen(['./train_mini' num2str(i) '.txt'], 'w');
    mte = fopen(['./val_mini' num2str(i) '.txt'], 'w');
    for j = 1:40000
        if Ctr{2}(j) == i - 1
            fprintf(mtr, '%s %d\n', char(Ctr{1}(j)), Ctr{3}(j));
        end
    end
    for j = 1:2500
        if Cte{2}(j) == i - 1
            fprintf(mte, '%s %d\n', char(Cte{1}(j)), Cte{3}(j));
        end
    end
    fclose(mtr);
    fclose(mte);
end

fclose(ftr);
fclose(fte);