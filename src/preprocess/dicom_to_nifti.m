% Convert Dicom to Nifti files
% from dicomdir to niftidir

fid = fopen('directory_list.txt');
fgetl(fid);
rootdir = fgetl(fid);
fclose(fid);

dataset_list = {'local_gbm', 'local_glioma', 'tcia_gbm', 'tcia_glioma'};
dataset = strcat(dataset_list(1), '/')

dicomdir = char(fullfile(rootdir, 'dicom/', dataset)); %inputdir
niftidir = char(fullfile(rootdir, 'nifti/', dataset)); %outputdir

%convert nii
dicomdir
nrun = 1; % enter the number of runs here
files = dir(strcat(dicomdir, '*'));

dir(niftidir)
dir(dicomdir)
files;
spm('defaults', 'FMRI');
spm_jobman('initcfg');
for id = 1:length(files)
    [pathstr, name, ext] = fileparts(files(id).name);
    %if length(name) < 4
    %    continue
    %end
    fullfile(dicomdir, name);
    if ~isfolder(fullfile(dicomdir, name))
        continue
    end
    if exist(fullfile(dicomdir, name)) ~= 7
        continue
    end
    fprintf('%s\n', name);
    fprintf(fullfile(niftidir, name));
    fprintf('\n')
    mkdir(fullfile(niftidir, name));
    ddir = fullfile(dicomdir, name, '*');
    dfiles = dir(ddir);
    dfiles=dfiles(~ismember({dfiles.name}, {'.', '..'}));
    dfiles.name;
    clear matlabbatch
    matlabbatch{1}.spm.util.import.dicom.data = cellstr(fullfile(dicomdir, name, {dfiles.name}')); % DICOM Import: Input files - cfg_files
    matlabbatch{1}.spm.util.import.dicom.outdir = cellstr(fullfile(niftidir, name)); % DICOM Import: Output directory - cfg_files
    matlabbatch{1}.spm.util.import.dicom.root = 'flat';
    matlabbatch{1}.spm.util.import.dicom.protfilter = '.*';
    matlabbatch{1}.spm.util.import.dicom.convopts.format = 'nii';
    matlabbatch{1}.spm.util.import.dicom.convopts.icedims = 0;
    try
        spm_jobman('run', matlabbatch)
    end

    file=dir(fullfile(maskodir, name, 's*.nii'));
    [m, n] = size(file);
    tfiles = {file.name};
    class(file)
    if m == 0
        continue
    end
    for j = 1:m
        ofile=fullfile(niftidir, name, strcat(int2str(j), '.nii'))
        ifile=fullfile(niftidir, name, tfiles{j})
        movefile(ifile, ofile);
    end
end
