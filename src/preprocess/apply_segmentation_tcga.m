% List of open inputs

fid = fopen('directory_list.txt');
fgetl(fid);
rootdir = fgetl(fid);
fgetl(fid);
scriptdir = fgetl(fid);
fgetl(fid);
tpmdir = fgetl(fid);
fclose(fid);

dataset_list = {'local_gbm', 'local_glioma', 'tcia_gbm', 'tcia_glioma'};
flag_dataset = dataset_list(3);
flag_dataset
maskdir = fullfile(rootdir, '/nifti/', flag_dataset)
types = {'GD', 'FLAIR', 'T2', 'T1', 'GT'};

nrun = 1; % enter the number of runs here

files=dir(maskdir{1});
spm('defaults', 'FMRI');
spm_jobman('initcfg');

flag = true;
for id = 1:length(files)
    [pathstr, name, ext] = fileparts(files(id).name)
    name
    %if length(name) ~= 4;
    %     continue
    %end
    if ~flag
        continue
    end
    if strcmp(name, '.');
        continue
    elseif strcmp(name, '..');
        continue
    end
    path = fullfile(maskdir, name)
    fprintf(1, '%s', name);
    fprintf(1, '%s', path{1});
    length(name);
    pathstr, name, ext;
    original_mni_types = {}
    for s = 1:length(types)
        subdir = types(s);
        original = fullfile(maskdir, name, strcat(subdir, '.nii'))
        if exist(original{1}, 'file') ~= 2
            continue
        end
        original_mni_types{end+1} = subdir{1}
    end
    if length(original_mni_types) == 0
        continue
    end
    clear matlabbatch
    if true
        for s = 1:length(original_mni_types)
            s
       %break
            def = fullfile(maskdir, name, strcat('y_', original_mni_types(s), '_changed.nii'))
            if exist(def{1}, 'file') ~= 2
                ;
            else
                continue
            end
            try
                cellstr(fullfile(maskdir, name, strcat(original_mni_types(s), '_changed.nii')))
                cellstr(fullfile(maskdir, name, strcat(original_mni_types(s), '.nii'))')
                matlabbatch{1}.spm.spatial.preproc.channel.vols = cellstr(fullfile(maskdir, name, strcat(original_mni_types(s), '_changed.nii'))')
                matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
                matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
                matlabbatch{1}.spm.spatial.preproc.channel.write = [0 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm = {fullfile(tpmdir, 'TPM.nii,1')};
                matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus = 1;
                matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [1 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [0 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm = {fullfile(tpmdir, 'TPM.nii,2')};
                matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus = 1;
                matlabbatch{1}.spm.spatial.preproc.tissue(2).native = [1 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = [0 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm = {fullfile(tpmdir, 'TPM.nii,3')};
                matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus = 2;
                matlabbatch{1}.spm.spatial.preproc.tissue(3).native = [1 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = [0 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(4).tpm = {fullfile(tpmdir, 'TPM.nii,4')};
                matlabbatch{1}.spm.spatial.preproc.tissue(4).ngaus = 3;
                matlabbatch{1}.spm.spatial.preproc.tissue(4).native = [1 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(4).warped = [0 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(5).tpm = {fullfile(tpmdir, 'TPM.nii,5')};
                matlabbatch{1}.spm.spatial.preproc.tissue(5).ngaus = 4;
                matlabbatch{1}.spm.spatial.preproc.tissue(5).native = [1 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(5).warped = [0 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(6).tpm = {fullfile(tpmdir, 'TPM.nii,6')};
                matlabbatch{1}.spm.spatial.preproc.tissue(6).ngaus = 2;
                matlabbatch{1}.spm.spatial.preproc.tissue(6).native = [0 0];
                matlabbatch{1}.spm.spatial.preproc.tissue(6).warped = [0 0];
                matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
                matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
                matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
                matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
                matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
                matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;
                matlabbatch{1}.spm.spatial.preproc.warp.write = [0 1];
                spm_jobman('run', matlabbatch)
            catch
                1
            end
        end
        % copyfile fullfile(maskdir, name, strcat(original_mni_types(s), '.nii')) fullfile(maskdir, name, strcat('2w',original_mni_types(s), '.nii'))
    end
    clear matlabbatch
    if true
        for s = 1:length(original_mni_types)
            subdir = original_mni_types(s)
            original = fullfile(maskdir, name, strcat(subdir, '_changed.nii'))
            def = fullfile(maskdir, name, strcat('y_', subdir, '_changed.nii'));
            original{1}
            if exist(original{1}, 'file') ~= 2
                aoeu
            end
            clear matlabbatch
            matlabbatch{1}.spm.spatial.normalise.write.subj = struct('def', {def(1)}, 'resample', {original(1)});
            matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
                                                                      78 76 85];
            % matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
            % matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'w';
            matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [1 1 1];
            matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = '2w';
            matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
            spm_jobman('run', matlabbatch)
        end
        msubdir = {'GD', 'GT'}
        for s = 1:length(msubdir)
            subdir = original_mni_types(1);
            def = fullfile(maskdir, name, strcat('y_', subdir, '_changed.nii'));
            mask = fullfile(maskdir, name, strcat(msubdir(s), '_mask_changed.nii'))
            out = fullfile(maskdir, name, strcat('2w', msubdir(s), '_mask.nii'))
            if exist(mask{1}, 'file') ~= 2
                'no mask'
                continue
            end
            clear matlabbatch
            matlabbatch{1}.spm.spatial.normalise.write.subj = struct('def', {def(1)}, 'resample', {mask(1)});
            matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
                                                                      78 76 85];
            matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [1 1 1];
            matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = '2w';
            matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
            spm_jobman('run', matlabbatch)
        end
    end
end
