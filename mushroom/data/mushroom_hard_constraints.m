%% Mushroom Hard Constraints
clear all; clc;

data_name                           = 'mushroom'; %adult/mammo/tictactoe
compname                            = 'berkmac';

%set directories
switch compname
    
    case 'berkmac'
        dropboxdir ='/Users/berk/Desktop/Dropbox/Research/';
        debugdir    = [dropboxdir, 'SLIM/MATLAB/Debug/'];
        codedir     = [dropboxdir, 'SLIM/MATLAB/'];
        datadir     = [dropboxdir, 'SLIM/Data/'];
        rawdatadir  = [datadir,'Raw Data Files/',data_name,'/'];
end

addpath(codedir);
cd(datadir);

%List All Data Files
data_file_names = dir([data_name,'*_processed.mat']);
data_file_names = {data_file_names(:).name};
data_long_names = regexprep(data_file_names,'_processed.mat','');

%Load Data File
data_processed_file  = data_file_names{1};
load(data_processed_file);

if ~exist('HardConstraints','var')
    HardConstraints = struct();
end





%% U001 (coefs between 20-20; L0_max = 10)

HardConstraints(1).Lset         = [];
HardConstraints(1).L0_min       = 0;
HardConstraints(1).L0_max       = NaN;
HardConstraints(1).epsilon      = 0.001;
HardConstraints(1).M            = NaN;
HardConstraints(1).err_min      = 0;
HardConstraints(1).err_max      = 1;
HardConstraints(1).pos_err_min  = 0;
HardConstraints(1).pos_err_max  = 1;
HardConstraints(1).neg_err_min  = 0;
HardConstraints(1).neg_err_max  = 1;


%% Create Individual Lset and MNSet and Save

for i = 1:length(data_file_names)
    
    data                        = load(data_file_names{i});
    data_name                   = regexprep(data_file_names{i},'_processed.mat','');
    
    %Create Lset
    s        = CreateEmptyLset();
    s.name   = 'default';
    s.class  = 'bounded';
    s.type   = 'I';
    s.lim    = [-10 10];
    
    s = CheckLset(s);
    
    s(2)=s(1);
    s(2).name      = '(Intercept)';
    s(2).type      = 'I';
    s(2).lim       = [-100 100];
    s(2).C_0j      = 0.00;
    
    Lset = CreateLset(data.X_headers,s);
    PrintLset(Lset)

    %Save Lset to all Hard Constraints
    for h = 1:length(HardConstraints)
        HardConstraints(h).Lset = Lset;
    end
    
    %Create MN set
    if isfield(data,'MNset')
        MNset = data.MNset;
    else
        MNset.X = data.X;
        MNset.X_headers = data.X_headers;
        MNset = CreateMNRLData(MNset);
    end
    
    %print for sanity
    MNset.X_headers(:)
    
    save(data_file_names{i},'HardConstraints','MNset','-append')
    constraint_file_name        = [datadir, data_name, '_hardconstraints_R.mat'];
    constraint_script_file      = [datadir, data_name, '_hardconstraints_script.command'];
    
    PassHardConstraintsToR(constraint_file_name,constraint_script_file,data_file_names{i},HardConstraints)
    
end




