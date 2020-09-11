function rc = dicom2nifti(input_dir)
    input_dir
    dirs = dir(char(input_dir));
    dirs = dirs(~startsWith({dirs.name},'.'));
    dirs = {dirs.name};

    for k=1:length(dirs)
        curdir = dirs{k};
        curdir = fullfile(char(input_dir), curdir)
        [files, ds] = spm_select('FPList', curdir, '.dcm$');
        spm_defaults;
        hdr = spm_dicom_headers(files);
        spm_dicom_convert(hdr, 'all', 'flat', 'nii', curdir);
    end
    display('done!')
    rc = 0;
end