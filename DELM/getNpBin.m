function np=getNpBin()
    if not (exist('./npbin')==7)
        error('no data dir!');
    end
    np=struct();
    lsStr = ls('npbin');
    lsStr = regexprep(lsStr,'\s{2,}','+');
    fileNameCell = regexp(lsStr,'+','split');
    fileNum = length(fileNameCell);
    for ii=1:fileNum
        eachFileName = fileNameCell{ii};
        eachFileName = deblank(eachFileName);
        fprintf('loding file %s\n', eachFileName)
        parts = regexp(eachFileName,'_','split');
        if length(parts)~=4
            fprintf('file name error of %s\n',eachFileName);
            continue
        end
 
        varName = parts{1};
        type = parts{3};
        shapeStr = parts{2};
        shapeCell = regexp(shapeStr, '-', 'split');
        n = length(shapeCell);
        switchArray = 1:n;
        temp = switchArray(1);
        switchArray(1) = switchArray(2);
        switchArray(2) = temp;
        switchStr = '[';
        for i=1:n
            switchStr = [switchStr,num2str(switchArray(i)),','];
        end
        switchStr = [switchStr,']'];
 
        shapeArray = zeros(1,n);
        for i=1:n
            shapeArray(i) = str2num(shapeCell{n-i+1});
        end
        fid=fopen(['./npbin/',eachFileName],'r');
        eval(['tempRead','=fread(fid,inf,''',type,''');']);
        fclose(fid);
        tempRead = reshape(tempRead,shapeArray);
        eval(['np.',varName,'=','permute(tempRead,',switchStr,');']);
        fprintf([varName,' loaded!!\n']);
    end
end

