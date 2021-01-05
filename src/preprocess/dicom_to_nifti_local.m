% List of open inputs
% This script removes all *.nii files in output directory.
% Directory: 4-digit patient id

fid = fopen('directory_list.txt');
fgetl(fid);
rootdir = fgetl(fid);
fgetl(fid);
scriptdir = fgetl(fid);
fgetl(fid);
tpmdir = fgetl(fid);
fclose(fid);

gbm = false;
if gbm
    dataset = 'local_gbm/';
else
    dataset = 'local_glioma/';
end
dicomdir = fullfile(rootdir, '/dicom/', dataset); %dicom directory
niftidir = fullfile(rootdir, '/nifti/', dataset); % mask input directory

first_id = '0001'
dicomdir
nrun = 1; % enter the number of runs here
files = dir(strcat(dicomdir, '*'));

mkdir(niftidir);
dir(niftidir)
spm('defaults', 'FMRI');
spm_jobman('initcfg');
flag = true;

for id = 1:length(files)
    [pathstr, name, ext] = fileparts(files(id).name);
    if length(name) < 4;
        continue
    end
    name
    if strcmp(name, first_id) % set the first patient id for restarting
        flag = true;
    end
    if ~flag
        continue
    end
    fprintf('%s', name);
    if gbm
        types = {'GD', 'FLAIR', 'T2', 'T1'};
    else
        types = {'GD', 'GT', 'FLAIR', 'T2', 'T1'};
    end
    fprintf(fullfile(niftidir, name));
    mkdir(fullfile(niftidir, name));
    delete(fullfile(niftidir, name, '*.nii'));
    pathstr, name, ext;
    for s = 1:length(types)
        subdir = types(s);
        if length(types) == 5 && s == 2
            subdir = {'GD'}; % Use GD dicom files for GT;
        end
        ddir = fullfile(dicomdir, name, subdir, '*');
        dfiles = dir(ddir{1});
        dfiles=dfiles(~ismember({dfiles.name}, {'.', '..'}));
        if length(dfiles) == 0
            continue
        end
        dfiles
        clear matlabbatch
        dfiles.name
        matlabbatch{1}.spm.util.import.dicom.data = cellstr(fullfile(dicomdir, name, subdir, {dfiles.name}')); % DICOM Import: Input files - cfg_files
        matlabbatch{1}.spm.util.import.dicom.outdir = cellstr(fullfile(niftidir, name)); % DICOM Import: Output directory - cfg_files
        matlabbatch{1}.spm.util.import.dicom.root = 'flat';
        matlabbatch{1}.spm.util.import.dicom.protfilter = '.*';
        matlabbatch{1}.spm.util.import.dicom.convopts.format = 'nii';
        matlabbatch{1}.spm.util.import.dicom.convopts.icedims = 0;
        spm_jobman('run', matlabbatch)
        file=dir(fullfile(niftidir, name, 's*.nii'));
        [m, n] = size(file)
        tfiles = {file.name}
        class(file)
        if m == 0
            continue
        end
        for j = 1:m
            if j == 1;
                ofile=fullfile(niftidir, name, strcat(types(s), '.nii'));
            else
                ofile=fullfile(niftidir, name, strcat(types(s), int2str(j), '.nii'));
            end
            movefile(fullfile(niftidir, name, tfiles{j}), ofile{1});
        end
    end
end
