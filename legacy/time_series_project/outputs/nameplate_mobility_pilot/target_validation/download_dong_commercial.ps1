$ErrorActionPreference = 'Stop'
$folder = Join-Path $PSScriptRoot 'direct_sources'
New-Item -ItemType Directory -Path $folder -Force | Out-Null
$uri = 'https://datafile.seoul.go.kr/bigfile/iot/sheet/csv/download.do'
$quarters = @('20231','20232','20233','20234','20241','20242','20251','20252')
$sources = @(
    @{ Name='footfall_dong'; Id='OA-22178' }
)
foreach ($source in $sources) {
    foreach ($quarter in $quarters) {
        $dest = Join-Path $folder "$($source.Name)_$quarter.csv"
        if (-not (Test-Path -LiteralPath $dest)) {
            $body = @{
                srvType='S'; infId=$source.Id; serviceKind='0'; pageNo='1';
                gridTotalCnt='30000'; ssUserId='SAMPLE_VIEW'; strWhere='';
                strOrderby='STDR_YYQU_CD DESC'; filterCol='STDR_YYQU_CD'; txtFilter=$quarter
            }
            Invoke-WebRequest -Uri $uri -Method Post -Body $body -Headers @{
                Referer="https://data.seoul.go.kr/dataList/$($source.Id)/S/1/datasetView.do"
            } -OutFile $dest
        }
        $item = Get-Item -LiteralPath $dest
        Write-Output "$($source.Name) $quarter $($item.Length) bytes"
    }
}
