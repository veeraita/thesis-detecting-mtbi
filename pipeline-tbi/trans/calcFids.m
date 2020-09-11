function rc = calcFids(idx)
    addpath(fullfile(spm('Dir'),'external','fieldtrip'));
    clear ft_defaults
    clear global ft_default
    ft_defaults;
    global ft_default
    ft_default.trackcallinfo = 'no';
    ft_default.showcallinfo = 'no';

    addpath(...
    fullfile(spm('Dir'),'external','bemcp'),...
    fullfile(spm('Dir'),'external','ctf'),...
    fullfile(spm('Dir'),'external','eeprobe'),...
    fullfile(spm('Dir'),'external','mne'),...
    fullfile(spm('Dir'),'toolbox', 'dcm_meeg'),...
    fullfile(spm('Dir'),'toolbox', 'spectral'),...
    fullfile(spm('Dir'),'toolbox', 'Neural_Models'),...
    fullfile(spm('Dir'),'toolbox', 'MEEGtools'));

  
    blockSize=10;
    input_dir = getenv('MRI_DIR')
    D = dir(input_dir);
    D = D(~startsWith({D.name},'.'));
    
    %idx = str2num(idx)
    for i = (blockSize*idx+1):min(((1+idx)*blockSize),numel(D))
        currSub = D(i).name
        getfid(currSub);
    end
    rc = 0;
end