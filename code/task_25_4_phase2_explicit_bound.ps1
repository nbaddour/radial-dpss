<#
Task 25.4, Phase 2: exact-rational audit of the completely explicit
missing-trace bound.

The displayed three-decimal bounds are certified by integer cross
multiplication. No floating-point value is used to decide an inequality.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-DeltaFraction {
    param(
        [Parameter(Mandatory = $true)][int]$Order,
        [Parameter(Mandatory = $true)][int]$BandwidthProduct,
        [Parameter(Mandatory = $true)][int]$Dimension
    )

    # b_P = 3(P + 3/4) = (12P + 9)/4.
    $bNumerator = [System.Numerics.BigInteger](12 * $Dimension + 9)
    $denominatorDifference =
        $bNumerator * $bNumerator -
        [System.Numerics.BigInteger](16 * $BandwidthProduct * $BandwidthProduct)

    if ($denominatorDifference -le 0) {
        throw "The explicit bound requires 3(P+3/4) > c."
    }

    # tau = b^2/(b^2-c^2)^2 + b/[3(b^2-c^2)]
    #     = 4 B (12 B + D)/(3 D^2),
    # where B=12P+9 and D=B^2-16c^2.
    if ($Order -eq 0) {
        # 2 sigma_0(c) = 4c/3.
        $numerator =
            [System.Numerics.BigInteger](16 * $BandwidthProduct) *
            $bNumerator *
            (12 * $bNumerator + $denominatorDifference)
        $denominator =
            [System.Numerics.BigInteger]9 *
            $denominatorDifference *
            $denominatorDifference
    }
    else {
        # 2 sigma_n(c) = (4c+4n^2-1)/2 for n >= 1.
        $numerator =
            [System.Numerics.BigInteger](2 * (
                4 * $BandwidthProduct + 4 * $Order * $Order - 1
            )) *
            $bNumerator *
            (12 * $bNumerator + $denominatorDifference)
        $denominator =
            [System.Numerics.BigInteger]3 *
            $denominatorDifference *
            $denominatorDifference
    }

    [pscustomobject]@{
        Numerator = $numerator
        Denominator = $denominator
    }
}

# P = ceil(c/pi)+20 for c = 10,20,40,80. The integer values below
# follow from standard rational bounds on pi.
$cases = @(
    [pscustomobject]@{ c = 10; P = 24 },
    [pscustomobject]@{ c = 20; P = 27 },
    [pscustomobject]@{ c = 40; P = 33 },
    [pscustomobject]@{ c = 80; P = 46 }
)

$reportedThousandths = @{
    '0,10' = 64;  '1,10' = 103; '2,10' = 131; '4,10' = 246
    '0,20' = 118; '1,20' = 184; '2,20' = 210; '4,20' = 316
    '0,40' = 216; '1,40' = 330; '2,40' = 354; '4,40' = 451
    '0,80' = 388; '1,80' = 587; '2,80' = 609; '4,80' = 697
}

$rows = foreach ($case in $cases) {
    foreach ($n in 0, 1, 2, 4) {
        $fraction = Get-DeltaFraction -Order $n -BandwidthProduct $case.c -Dimension $case.P

        $key = "$n,$($case.c)"
        $upperThousandths = [int]$reportedThousandths[$key]

        # Prove numerator/denominator < upperThousandths/1000.
        $margin =
            [System.Numerics.BigInteger]$upperThousandths *
            $fraction.Denominator -
            [System.Numerics.BigInteger]1000 *
            $fraction.Numerator

        if ($margin -le 0) {
            throw "Reported upper bound failed exact cross-product check for $key."
        }

        [pscustomobject]@{
            n = $n
            c = $case.c
            P = $case.P
            CertifiedUpperBound = '{0:N3}' -f ($upperThousandths / 1000)
            IntegerMarginPositive = $true
        }
    }
}

$rows | Format-Table -AutoSize

