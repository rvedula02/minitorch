# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py


MNIST_TRAINING:
Epoch 1 (batch 1) loss 2.3096511196505958 valid acc 2/16 best valid acc 2
Epoch 1 (batch 6) loss 11.541347388440036 valid acc 2/16 best valid acc 2
Epoch 1 (batch 11) loss 11.499474441040682 valid acc 3/16 best valid acc 3
Epoch 1 (batch 16) loss 11.435423874043309 valid acc 4/16 best valid acc 4
Epoch 1 (batch 21) loss 11.501187809475685 valid acc 1/16 best valid acc 4
Epoch 1 (batch 26) loss 11.265490106875664 valid acc 3/16 best valid acc 4
Epoch 1 (batch 31) loss 11.740804034114335 valid acc 4/16 best valid acc 4
Epoch 1 (batch 36) loss 11.060970587361552 valid acc 5/16 best valid acc 5
Epoch 1 (batch 41) loss 11.043221365516292 valid acc 5/16 best valid acc 5
Epoch 1 (batch 46) loss 10.609301601826486 valid acc 6/16 best valid acc 6
Epoch 1 (batch 51) loss 10.99579134384298 valid acc 5/16 best valid acc 6
Epoch 1 (batch 56) loss 10.565224318460015 valid acc 5/16 best valid acc 6
Epoch 1 (batch 61) loss 10.425501020733293 valid acc 7/16 best valid acc 7
Epoch 1 (batch 66) loss 9.511440573340895 valid acc 5/16 best valid acc 7
Epoch 1 (batch 71) loss 9.402363738587132 valid acc 8/16 best valid acc 8
Epoch 1 (batch 76) loss 8.264621832494273 valid acc 7/16 best valid acc 8
Epoch 1 (batch 81) loss 8.58630779523768 valid acc 8/16 best valid acc 8
Epoch 1 (batch 86) loss 8.604658443218431 valid acc 10/16 best valid acc 10
Epoch 1 (batch 91) loss 8.395032417680609 valid acc 10/16 best valid acc 10
Epoch 1 (batch 96) loss 7.355117425459903 valid acc 10/16 best valid acc 10
Epoch 1 (batch 101) loss 7.070017852742682 valid acc 8/16 best valid acc 10
Epoch 1 (batch 106) loss 6.010543564898192 valid acc 10/16 best valid acc 10
Epoch 1 (batch 111) loss 5.632636485787838 valid acc 10/16 best valid acc 10
Epoch 1 (batch 116) loss 6.092751542155073 valid acc 10/16 best valid acc 10
Epoch 1 (batch 121) loss 6.538909723059575 valid acc 10/16 best valid acc 10
Epoch 1 (batch 126) loss 5.9064394446971855 valid acc 12/16 best valid acc 12
Epoch 1 (batch 131) loss 6.542955185636074 valid acc 9/16 best valid acc 12
Epoch 1 (batch 136) loss 5.610340447545749 valid acc 12/16 best valid acc 12
Epoch 1 (batch 141) loss 5.129148666893087 valid acc 14/16 best valid acc 14
Epoch 1 (batch 146) loss 4.411928633783639 valid acc 12/16 best valid acc 14
Epoch 1 (batch 151) loss 6.354340249167105 valid acc 10/16 best valid acc 14
Epoch 1 (batch 156) loss 5.749885458115385 valid acc 11/16 best valid acc 14
Epoch 1 (batch 161) loss 6.182901983911684 valid acc 7/16 best valid acc 14
Epoch 1 (batch 166) loss 6.3856557274986745 valid acc 13/16 best valid acc 14
Epoch 1 (batch 171) loss 6.657150078476378 valid acc 12/16 best valid acc 14
Epoch 1 (batch 176) loss 5.633481542479734 valid acc 10/16 best valid acc 14
Epoch 1 (batch 181) loss 4.260853775755631 valid acc 13/16 best valid acc 14
Epoch 1 (batch 186) loss 4.910228799306874 valid acc 11/16 best valid acc 14
Epoch 1 (batch 191) loss 4.576929363103008 valid acc 15/16 best valid acc 15
Epoch 1 (batch 196) loss 4.867256439036743 valid acc 12/16 best valid acc 15
Epoch 1 (batch 201) loss 4.132623559061282 valid acc 12/16 best valid acc 15
Epoch 1 (batch 206) loss 4.141821394656973 valid acc 15/16 best valid acc 15
Epoch 1 (batch 211) loss 4.8725414390676685 valid acc 11/16 best valid acc 15
Epoch 1 (batch 216) loss 4.321242862034812 valid acc 12/16 best valid acc 15
Epoch 1 (batch 221) loss 5.010852924154424 valid acc 13/16 best valid acc 15
Epoch 1 (batch 226) loss 3.690592007522367 valid acc 11/16 best valid acc 15
Epoch 1 (batch 231) loss 4.754226673473591 valid acc 12/16 best valid acc 15
Epoch 1 (batch 236) loss 4.68376421961111 valid acc 10/16 best valid acc 15
Epoch 1 (batch 241) loss 3.1185780497605244 valid acc 12/16 best valid acc 15
Epoch 1 (batch 246) loss 3.2395327183238045 valid acc 11/16 best valid acc 15
Epoch 1 (batch 251) loss 3.069977446780575 valid acc 12/16 best valid acc 15
Epoch 1 (batch 256) loss 4.658309775087918 valid acc 12/16 best valid acc 15
Epoch 1 (batch 261) loss 3.4527521355892095 valid acc 10/16 best valid acc 15
Epoch 1 (batch 266) loss 6.711916271870168 valid acc 11/16 best valid acc 15
Epoch 1 (batch 271) loss 9.376289921385787 valid acc 11/16 best valid acc 15
Epoch 1 (batch 276) loss 4.632113686309927 valid acc 12/16 best valid acc 15
Epoch 1 (batch 281) loss 4.421750983254919 valid acc 12/16 best valid acc 15
Epoch 1 (batch 286) loss 3.663256702221157 valid acc 13/16 best valid acc 15
Epoch 1 (batch 291) loss 3.9811102922150408 valid acc 13/16 best valid acc 15
Epoch 1 (batch 296) loss 3.902849422101834 valid acc 12/16 best valid acc 15
Epoch 1 (batch 301) loss 4.537657706231909 valid acc 14/16 best valid acc 15
Epoch 1 (batch 306) loss 3.650713995584062 valid acc 13/16 best valid acc 15
Epoch 1 (batch 311) loss 5.144104506360884 valid acc 13/16 best valid acc 15
Epoch 2 (batch 313) loss 0.31122089741466297 valid acc 12/16 best valid acc 15
Epoch 2 (batch 318) loss 3.9319519225772335 valid acc 14/16 best valid acc 15
Epoch 2 (batch 323) loss 3.6035288380990176 valid acc 13/16 best valid acc 15
Epoch 2 (batch 328) loss 4.449567898781669 valid acc 10/16 best valid acc 15
Epoch 2 (batch 333) loss 7.870714050205324 valid acc 11/16 best valid acc 15
Epoch 2 (batch 338) loss 4.516725192506275 valid acc 12/16 best valid acc 15
Epoch 2 (batch 343) loss 4.053493769968975 valid acc 11/16 best valid acc 15
Epoch 2 (batch 348) loss 4.688451040620588 valid acc 12/16 best valid acc 15
Epoch 2 (batch 353) loss 3.923612779763414 valid acc 13/16 best valid acc 15
Epoch 2 (batch 358) loss 10.04004984640579 valid acc 4/16 best valid acc 15
Epoch 2 (batch 363) loss 18.76629835363344 valid acc 10/16 best valid acc 15
Epoch 2 (batch 368) loss 5.572175553198171 valid acc 13/16 best valid acc 15
Epoch 2 (batch 373) loss 4.339301718024454 valid acc 14/16 best valid acc 15
Epoch 2 (batch 378) loss 4.905512049451498 valid acc 14/16 best valid acc 15
Epoch 2 (batch 383) loss 4.558593717966607 valid acc 11/16 best valid acc 15
Epoch 2 (batch 388) loss 5.775421137816026 valid acc 12/16 best valid acc 15
Epoch 2 (batch 393) loss 5.797353839144457 valid acc 10/16 best valid acc 15
Epoch 2 (batch 398) loss 4.554455354043355 valid acc 10/16 best valid acc 15
Epoch 2 (batch 403) loss 3.992972127963974 valid acc 13/16 best valid acc 15
Epoch 2 (batch 408) loss 5.024890491637169 valid acc 11/16 best valid acc 15
Epoch 2 (batch 413) loss 4.27825934216502 valid acc 12/16 best valid acc 15
Epoch 2 (batch 418) loss 3.484724529935968 valid acc 11/16 best valid acc 15
Epoch 2 (batch 423) loss 2.443924080144 valid acc 13/16 best valid acc 15
Epoch 2 (batch 428) loss 4.13175334643026 valid acc 12/16 best valid acc 15
Epoch 2 (batch 433) loss 2.7938520949322365 valid acc 14/16 best valid acc 15
Epoch 2 (batch 438) loss 3.0587490722449524 valid acc 13/16 best valid acc 15
Epoch 2 (batch 443) loss 3.813339393469777 valid acc 13/16 best valid acc 15
Epoch 2 (batch 448) loss 3.0965256098673817 valid acc 14/16 best valid acc 15
Epoch 2 (batch 453) loss 1.9488363024981332 valid acc 13/16 best valid acc 15
Epoch 2 (batch 458) loss 2.080916936801722 valid acc 12/16 best valid acc 15
Epoch 2 (batch 463) loss 3.1250303620917954 valid acc 13/16 best valid acc 15
Epoch 2 (batch 468) loss 3.8311743113073446 valid acc 13/16 best valid acc 15
Epoch 2 (batch 473) loss 2.420715262234432 valid acc 13/16 best valid acc 15
Epoch 2 (batch 478) loss 3.19036961842988 valid acc 13/16 best valid acc 15
Epoch 2 (batch 483) loss 4.22462917569927 valid acc 12/16 best valid acc 15
Epoch 2 (batch 488) loss 4.360420145817647 valid acc 12/16 best valid acc 15
Epoch 2 (batch 493) loss 2.889720702329071 valid acc 12/16 best valid acc 15
Epoch 2 (batch 498) loss 2.77450005572164 valid acc 12/16 best valid acc 15
Epoch 2 (batch 503) loss 2.961824843606491 valid acc 13/16 best valid acc 15
Epoch 2 (batch 508) loss 3.371428428896619 valid acc 12/16 best valid acc 15
Epoch 2 (batch 513) loss 3.043968133093881 valid acc 13/16 best valid acc 15
Epoch 2 (batch 518) loss 3.322328151354684 valid acc 14/16 best valid acc 15
Epoch 2 (batch 523) loss 3.1387826403532357 valid acc 13/16 best valid acc 15
Epoch 2 (batch 528) loss 2.302458611526612 valid acc 13/16 best valid acc 15
Epoch 2 (batch 533) loss 3.4886319742756404 valid acc 13/16 best valid acc 15
Epoch 2 (batch 538) loss 2.043031644457003 valid acc 14/16 best valid acc 15
Epoch 2 (batch 543) loss 2.930914907368105 valid acc 14/16 best valid acc 15
Epoch 2 (batch 548) loss 2.8595617937661784 valid acc 12/16 best valid acc 15
Epoch 2 (batch 553) loss 2.1833848487635676 valid acc 11/16 best valid acc 15
Epoch 2 (batch 558) loss 1.7135824024825008 valid acc 11/16 best valid acc 15
Epoch 2 (batch 563) loss 3.3532096390775132 valid acc 13/16 best valid acc 15
Epoch 2 (batch 568) loss 3.089301910627729 valid acc 11/16 best valid acc 15
Epoch 2 (batch 573) loss 3.6851809068276236 valid acc 12/16 best valid acc 15
Epoch 2 (batch 578) loss 2.319766239666871 valid acc 12/16 best valid acc 15
Epoch 2 (batch 583) loss 2.898063164543614 valid acc 11/16 best valid acc 15
Epoch 2 (batch 588) loss 3.7585379190742234 valid acc 10/16 best valid acc 15
Epoch 2 (batch 593) loss 3.2406483181206154 valid acc 13/16 best valid acc 15
Epoch 2 (batch 598) loss 3.335151966954645 valid acc 10/16 best valid acc 15
Epoch 2 (batch 603) loss 3.739601585241357 valid acc 11/16 best valid acc 15
Epoch 2 (batch 608) loss 3.49465767181608 valid acc 13/16 best valid acc 15
Epoch 2 (batch 613) loss 2.3584102859654026 valid acc 13/16 best valid acc 15
Epoch 2 (batch 618) loss 2.427553449822615 valid acc 13/16 best valid acc 15
Epoch 2 (batch 623) loss 3.2098979835786516 valid acc 13/16 best valid acc 15
Epoch 3 (batch 625) loss 0.2900813028500143 valid acc 12/16 best valid acc 15
Epoch 3 (batch 630) loss 2.8127560956913737 valid acc 14/16 best valid acc 15
Epoch 3 (batch 635) loss 2.637030265349355 valid acc 12/16 best valid acc 15
Epoch 3 (batch 640) loss 3.318154509814484 valid acc 13/16 best valid acc 15
Epoch 3 (batch 645) loss 1.5677909881393939 valid acc 13/16 best valid acc 15
Epoch 3 (batch 650) loss 1.4678146130415741 valid acc 14/16 best valid acc 15
Epoch 3 (batch 655) loss 3.989951633489426 valid acc 11/16 best valid acc 15
Epoch 3 (batch 660) loss 4.557825037443967 valid acc 14/16 best valid acc 15
Epoch 3 (batch 665) loss 8.381618268640683 valid acc 11/16 best valid acc 15
Epoch 3 (batch 670) loss 3.280716299759154 valid acc 13/16 best valid acc 15
Epoch 3 (batch 675) loss 3.7946722782010744 valid acc 13/16 best valid acc 15
Epoch 3 (batch 680) loss 3.8971274352803915 valid acc 13/16 best valid acc 15
Epoch 3 (batch 685) loss 3.292445760638712 valid acc 13/16 best valid acc 15
Epoch 3 (batch 690) loss 4.512776312349903 valid acc 14/16 best valid acc 15
Epoch 3 (batch 695) loss 3.414302518912446 valid acc 12/16 best valid acc 15
Epoch 3 (batch 700) loss 3.3917429099809406 valid acc 12/16 best valid acc 15
Epoch 3 (batch 705) loss 3.4272655679472885 valid acc 13/16 best valid acc 15
Epoch 3 (batch 710) loss 3.3271569532461487 valid acc 15/16 best valid acc 15
Epoch 3 (batch 715) loss 2.3558562597034474 valid acc 13/16 best valid acc 15
Epoch 3 (batch 720) loss 3.225907640525717 valid acc 13/16 best valid acc 15
Epoch 3 (batch 725) loss 2.0468682256469237 valid acc 11/16 best valid acc 15
Epoch 3 (batch 730) loss 2.8747622950861844 valid acc 12/16 best valid acc 15
Epoch 3 (batch 735) loss 1.9811478436955694 valid acc 14/16 best valid acc 15
Epoch 3 (batch 740) loss 9.375719729141291 valid acc 12/16 best valid acc 15
Epoch 3 (batch 745) loss 3.3406706880918087 valid acc 11/16 best valid acc 15
Epoch 3 (batch 750) loss 3.2649954653684023 valid acc 12/16 best valid acc 15
Epoch 3 (batch 755) loss 3.1214770380625785 valid acc 12/16 best valid acc 15
Epoch 3 (batch 760) loss 2.109014424604752 valid acc 12/16 best valid acc 15
Epoch 3 (batch 765) loss 2.677309128688644 valid acc 13/16 best valid acc 15
Epoch 3 (batch 770) loss 1.6579427787046668 valid acc 13/16 best valid acc 15
Epoch 3 (batch 775) loss 3.8945872065991 valid acc 12/16 best valid acc 15
Epoch 3 (batch 780) loss 3.2547281861679402 valid acc 13/16 best valid acc 15
Epoch 3 (batch 785) loss 1.5469415181936363 valid acc 11/16 best valid acc 15
Epoch 3 (batch 790) loss 2.5554833281336764 valid acc 14/16 best valid acc 15
Epoch 3 (batch 795) loss 4.270974716586063 valid acc 14/16 best valid acc 15
Epoch 3 (batch 800) loss 4.153567154517762 valid acc 11/16 best valid acc 15
Epoch 3 (batch 805) loss 2.5038825434815895 valid acc 10/16 best valid acc 15
Epoch 3 (batch 810) loss 2.8817381224147365 valid acc 13/16 best valid acc 15
Epoch 3 (batch 815) loss 2.430079728920181 valid acc 12/16 best valid acc 15
Epoch 3 (batch 820) loss 1.8141268419688643 valid acc 13/16 best valid acc 15
Epoch 3 (batch 825) loss 2.4544617285796004 valid acc 12/16 best valid acc 15
Epoch 3 (batch 830) loss 2.459131939489866 valid acc 14/16 best valid acc 15
Epoch 3 (batch 835) loss 2.9655997995654984 valid acc 12/16 best valid acc 15
Epoch 3 (batch 840) loss 2.5674173997903513 valid acc 13/16 best valid acc 15
Epoch 3 (batch 845) loss 3.6426482147471626 valid acc 14/16 best valid acc 15
Epoch 3 (batch 850) loss 2.7295885847550085 valid acc 13/16 best valid acc 15
Epoch 3 (batch 855) loss 3.3506894940191367 valid acc 12/16 best valid acc 15
Epoch 3 (batch 860) loss 3.034396459015448 valid acc 15/16 best valid acc 15
Epoch 3 (batch 865) loss 2.50712405304742 valid acc 12/16 best valid acc 15
Epoch 3 (batch 870) loss 2.0799708802774033 valid acc 12/16 best valid acc 15
Epoch 3 (batch 875) loss 2.243321746454989 valid acc 13/16 best valid acc 15
Epoch 3 (batch 880) loss 2.145962404079969 valid acc 15/16 best valid acc 15
Epoch 3 (batch 885) loss 2.7070901902626394 valid acc 13/16 best valid acc 15
Epoch 3 (batch 890) loss 1.963966664391485 valid acc 14/16 best valid acc 15
Epoch 3 (batch 895) loss 2.8126492823294083 valid acc 13/16 best valid acc 15
Epoch 3 (batch 900) loss 2.7102782286433964 valid acc 12/16 best valid acc 15
Epoch 3 (batch 905) loss 3.3145898632501596 valid acc 12/16 best valid acc 15
Epoch 3 (batch 910) loss 1.5893904010889837 valid acc 13/16 best valid acc 15
Epoch 3 (batch 915) loss 2.0952369236963935 valid acc 14/16 best valid acc 15
Epoch 3 (batch 920) loss 2.8410504948206188 valid acc 12/16 best valid acc 15
Epoch 3 (batch 925) loss 2.884390207600219 valid acc 13/16 best valid acc 15
Epoch 3 (batch 930) loss 2.8314951625235656 valid acc 14/16 best valid acc 15
Epoch 3 (batch 935) loss 2.3568954053262257 valid acc 10/16 best valid acc 15
Epoch 4 (batch 937) loss 0.08099066762610535 valid acc 12/16 best valid acc 15
Epoch 4 (batch 942) loss 3.1480337975942954 valid acc 14/16 best valid acc 15
Epoch 4 (batch 947) loss 2.0887347430570293 valid acc 14/16 best valid acc 15
Epoch 4 (batch 952) loss 2.4442209524049976 valid acc 12/16 best valid acc 15
Epoch 4 (batch 957) loss 1.8094586121497545 valid acc 13/16 best valid acc 15
Epoch 4 (batch 962) loss 1.5909057973662426 valid acc 14/16 best valid acc 15
Epoch 4 (batch 967) loss 1.806368408005645 valid acc 13/16 best valid acc 15
Epoch 4 (batch 972) loss 2.9367685831500983 valid acc 14/16 best valid acc 15
Epoch 4 (batch 977) loss 3.7665315736174168 valid acc 12/16 best valid acc 15
Epoch 4 (batch 982) loss 1.640734272817422 valid acc 12/16 best valid acc 15
Epoch 4 (batch 987) loss 1.9852683454897995 valid acc 14/16 best valid acc 15
Epoch 4 (batch 992) loss 3.067390454504523 valid acc 14/16 best valid acc 15
Epoch 4 (batch 997) loss 3.568635379342606 valid acc 12/16 best valid acc 15
Epoch 4 (batch 1002) loss 3.864876658055653 valid acc 13/16 best valid acc 15
Epoch 4 (batch 1007) loss 3.0007235434343644 valid acc 14/16 best valid acc 15
Epoch 4 (batch 1012) loss 2.7102291044142643 valid acc 13/16 best valid acc 15
Epoch 4 (batch 1017) loss 3.735404619521488 valid acc 13/16 best valid acc 15
Epoch 4 (batch 1022) loss 3.1534638169490794 valid acc 12/16 best valid acc 15
Epoch 4 (batch 1027) loss 3.1308510737172814 valid acc 15/16 best valid acc 15
Epoch 4 (batch 1032) loss 3.1625750317841885 valid acc 13/16 best valid acc 15
Epoch 4 (batch 1037) loss 1.9143289291616852 valid acc 11/16 best valid acc 15
Epoch 4 (batch 1042) loss 1.5870422469415968 valid acc 15/16 best valid acc 15
Epoch 4 (batch 1047) loss 1.028873803033606 valid acc 13/16 best valid acc 15
Epoch 4 (batch 1052) loss 1.7564966636918005 valid acc 12/16 best valid acc 15
Epoch 4 (batch 1057) loss 2.6660949574000274 valid acc 13/16 best valid acc 15
Epoch 4 (batch 1062) loss 3.2261478187531942 valid acc 11/16 best valid acc 15
Epoch 4 (batch 1067) loss 2.050657454966588 valid acc 11/16 best valid acc 15
Epoch 4 (batch 1072) loss 3.763931543099641 valid acc 13/16 best valid acc 15
Epoch 4 (batch 1077) loss 2.614404709793888 valid acc 8/16 best valid acc 15
Epoch 4 (batch 1082) loss 2.9619450973884303 valid acc 9/16 best valid acc 15
Epoch 4 (batch 1087) loss 4.7259543859006925 valid acc 11/16 best valid acc 15
Epoch 4 (batch 1092) loss 4.892387659531574 valid acc 14/16 best valid acc 15
Epoch 4 (batch 1097) loss 2.3048882290302397 valid acc 14/16 best valid acc 15
Epoch 4 (batch 1102) loss 1.5361668053345638 valid acc 13/16 best valid acc 15
Epoch 4 (batch 1107) loss 3.8445975038324542 valid acc 14/16 best valid acc 15
Epoch 4 (batch 1112) loss 2.3975666703500864 valid acc 13/16 best valid acc 15
Epoch 4 (batch 1117) loss 2.572749770821541 valid acc 15/16 best valid acc 15
Epoch 4 (batch 1122) loss 2.1195264560133515 valid acc 14/16 best valid acc 15
Epoch 4 (batch 1127) loss 2.435658055854924 valid acc 15/16 best valid acc 15
Epoch 4 (batch 1132) loss 1.9549490412287702 valid acc 12/16 best valid acc 15
Epoch 4 (batch 1137) loss 2.333679261322678 valid acc 13/16 best valid acc 15
Epoch 4 (batch 1142) loss 1.4091460141820868 valid acc 13/16 best valid acc 15
Epoch 4 (batch 1147) loss 1.9293947544869234 valid acc 14/16 best valid acc 15
Epoch 4 (batch 1152) loss 2.237605084191796 valid acc 15/16 best valid acc 15
Epoch 4 (batch 1157) loss 2.454509157137182 valid acc 14/16 best valid acc 15
Epoch 4 (batch 1162) loss 2.27082219104222 valid acc 14/16 best valid acc 15
Epoch 4 (batch 1167) loss 2.743596415715848 valid acc 12/16 best valid acc 15
Epoch 4 (batch 1172) loss 2.2417894966105685 valid acc 14/16 best valid acc 15
Epoch 4 (batch 1177) loss 2.0492663457000337 valid acc 12/16 best valid acc 15
Epoch 4 (batch 1182) loss 1.2401500357552262 valid acc 13/16 best valid acc 15
Epoch 4 (batch 1187) loss 1.8274751873235275 valid acc 12/16 best valid acc 15
Epoch 4 (batch 1192) loss 1.4962516559743344 valid acc 13/16 best valid acc 15
Epoch 4 (batch 1197) loss 2.041653852047779 valid acc 11/16 best valid acc 15
Epoch 4 (batch 1202) loss 1.920333247334236 valid acc 13/16 best valid acc 15
Epoch 4 (batch 1207) loss 3.812282401036536 valid acc 14/16 best valid acc 15
Epoch 4 (batch 1212) loss 3.575685913161801 valid acc 13/16 best valid acc 15
Epoch 4 (batch 1217) loss 3.0409264250956998 valid acc 13/16 best valid acc 15
Epoch 4 (batch 1222) loss 2.648833599198084 valid acc 13/16 best valid acc 15
Epoch 4 (batch 1227) loss 4.103848821505746 valid acc 11/16 best valid acc 15
Epoch 4 (batch 1232) loss 5.032304916728361 valid acc 13/16 best valid acc 15
Epoch 4 (batch 1237) loss 2.4910240519040276 valid acc 15/16 best valid acc 15
Epoch 4 (batch 1242) loss 2.8476754937676056 valid acc 15/16 best valid acc 15
Epoch 4 (batch 1247) loss 2.311201602817097 valid acc 15/16 best valid acc 15
Epoch 5 (batch 1249) loss 0.18127449166540188 valid acc 13/16 best valid acc 15
Epoch 5 (batch 1254) loss 2.6953432753930002 valid acc 14/16 best valid acc 15
Epoch 5 (batch 1259) loss 1.2753510882032455 valid acc 13/16 best valid acc 15
Epoch 5 (batch 1264) loss 2.8479832575325603 valid acc 12/16 best valid acc 15
Epoch 5 (batch 1269) loss 1.6509011902121296 valid acc 15/16 best valid acc 15
Epoch 5 (batch 1274) loss 1.4055358275610725 valid acc 14/16 best valid acc 15
Epoch 5 (batch 1279) loss 2.271290232628607 valid acc 13/16 best valid acc 15
Epoch 5 (batch 1284) loss 2.647311008707773 valid acc 13/16 best valid acc 15
Epoch 5 (batch 1289) loss 2.0685466420836898 valid acc 15/16 best valid acc 15
Epoch 5 (batch 1294) loss 1.458935045155256 valid acc 14/16 best valid acc 15
Epoch 5 (batch 1299) loss 1.9274271600778665 valid acc 13/16 best valid acc 15
Epoch 5 (batch 1304) loss 3.204684240609905 valid acc 13/16 best valid acc 15
Epoch 5 (batch 1309) loss 2.9367863644008465 valid acc 12/16 best valid acc 15
Epoch 5 (batch 1314) loss 3.386117495072038 valid acc 13/16 best valid acc 15
Epoch 5 (batch 1319) loss 3.2179762141664203 valid acc 12/16 best valid acc 15
Epoch 5 (batch 1324) loss 1.8878710602686217 valid acc 14/16 best valid acc 15
Epoch 5 (batch 1329) loss 2.4966844892425857 valid acc 15/16 best valid acc 15
Epoch 5 (batch 1334) loss 2.9561762130898765 valid acc 12/16 best valid acc 15
Epoch 5 (batch 1339) loss 1.8337569279698158 valid acc 14/16 best valid acc 15
Epoch 5 (batch 1344) loss 2.7585601392763404 valid acc 14/16 best valid acc 15
Epoch 5 (batch 1349) loss 1.4705377315160713 valid acc 14/16 best valid acc 15
Epoch 5 (batch 1354) loss 1.2415577401909768 valid acc 13/16 best valid acc 15
Epoch 5 (batch 1359) loss 0.9926670718500044 valid acc 14/16 best valid acc 15
Epoch 5 (batch 1364) loss 1.4335063666897405 valid acc 14/16 best valid acc 15
Epoch 5 (batch 1369) loss 1.6054595388499622 valid acc 15/16 best valid acc 15
Epoch 5 (batch 1374) loss 1.927244422443027 valid acc 14/16 best valid acc 15
Epoch 5 (batch 1379) loss 1.7940774241988833 valid acc 16/16 best valid acc 16
Epoch 5 (batch 1384) loss 2.945923067413498 valid acc 14/16 best valid acc 16
Epoch 5 (batch 1389) loss 1.90575069248575 valid acc 14/16 best valid acc 16
Epoch 5 (batch 1394) loss 1.2014742258882722 valid acc 15/16 best valid acc 16
Epoch 5 (batch 1399) loss 1.800467242246805 valid acc 14/16 best valid acc 16
Epoch 5 (batch 1404) loss 2.9505484555316195 valid acc 13/16 best valid acc 16
Epoch 5 (batch 1409) loss 1.4792829859071999 valid acc 13/16 best valid acc 16
Epoch 5 (batch 1414) loss 1.313959959557952 valid acc 16/16 best valid acc 16
Epoch 5 (batch 1419) loss 3.5681193516401453 valid acc 15/16 best valid acc 16
Epoch 5 (batch 1424) loss 1.6409608340515165 valid acc 15/16 best valid acc 16
Epoch 5 (batch 1429) loss 2.139288470531896 valid acc 13/16 best valid acc 16
Epoch 5 (batch 1434) loss 1.6579220777160724 valid acc 13/16 best valid acc 16
Epoch 5 (batch 1439) loss 1.4953008761553266 valid acc 15/16 best valid acc 16
Epoch 5 (batch 1444) loss 2.0407611804693766 valid acc 13/16 best valid acc 16
Epoch 5 (batch 1449) loss 1.2755257235359292 valid acc 14/16 best valid acc 16
Epoch 5 (batch 1454) loss 2.345176793320759 valid acc 12/16 best valid acc 16
Epoch 5 (batch 1459) loss 1.9746393010728518 valid acc 13/16 best valid acc 16
Epoch 5 (batch 1464) loss 2.0601108186953225 valid acc 14/16 best valid acc 16
Epoch 5 (batch 1469) loss 3.122978016034549 valid acc 14/16 best valid acc 16
Epoch 5 (batch 1474) loss 2.6016446829171707 valid acc 14/16 best valid acc 16
Epoch 5 (batch 1479) loss 2.248176942756828 valid acc 15/16 best valid acc 16
Epoch 5 (batch 1484) loss 2.5042252490600756 valid acc 15/16 best valid acc 16
Epoch 5 (batch 1489) loss 1.6242305243359365 valid acc 14/16 best valid acc 16
Epoch 5 (batch 1494) loss 1.069387695534505 valid acc 15/16 best valid acc 16
Epoch 5 (batch 1499) loss 1.5407207653930688 valid acc 12/16 best valid acc 16
Epoch 5 (batch 1504) loss 2.57850772553855 valid acc 15/16 best valid acc 16
Epoch 5 (batch 1509) loss 2.767100192607707 valid acc 14/16 best valid acc 16
Epoch 5 (batch 1514) loss 2.3449858828407693 valid acc 15/16 best valid acc 16
Epoch 5 (batch 1519) loss 2.1719465558677884 valid acc 16/16 best valid acc 16
Epoch 5 (batch 1524) loss 1.6240892070512738 valid acc 13/16 best valid acc 16
Epoch 5 (batch 1529) loss 1.122665172338536 valid acc 16/16 best valid acc 16
Epoch 5 (batch 1534) loss 1.8928005261717264 valid acc 16/16 best valid acc 16
Epoch 5 (batch 1539) loss 1.898895143747085 valid acc 14/16 best valid acc 16
Epoch 5 (batch 1544) loss 2.341055590433168 valid acc 11/16 best valid acc 16
Epoch 5 (batch 1549) loss 1.9752743761658804 valid acc 14/16 best valid acc 16
Epoch 5 (batch 1554) loss 1.8820379263173048 valid acc 13/16 best valid acc 16
Epoch 5 (batch 1559) loss 2.286076924814705 valid acc 13/16 best valid acc 16
Epoch 6 (batch 1561) loss 0.02431955471487114 valid acc 13/16 best valid acc 16
Epoch 6 (batch 1566) loss 3.126221255892979 valid acc 12/16 best valid acc 16
Epoch 6 (batch 1571) loss 2.2390595995741034 valid acc 13/16 best valid acc 16
Epoch 6 (batch 1576) loss 1.3661430010379372 valid acc 12/16 best valid acc 16
Epoch 6 (batch 1581) loss 1.6091746210566007 valid acc 12/16 best valid acc 16
Epoch 6 (batch 1586) loss 1.0773968635078264 valid acc 15/16 best valid acc 16
Epoch 6 (batch 1591) loss 2.3281784194733612 valid acc 12/16 best valid acc 16
Epoch 6 (batch 1596) loss 5.565046152686196 valid acc 11/16 best valid acc 16
Epoch 6 (batch 1601) loss 2.3833524938186823 valid acc 13/16 best valid acc 16
Epoch 6 (batch 1606) loss 2.40097048421565 valid acc 12/16 best valid acc 16
Epoch 6 (batch 1611) loss 1.6316494460341764 valid acc 14/16 best valid acc 16
Epoch 6 (batch 1616) loss 3.2506038143841867 valid acc 12/16 best valid acc 16
Epoch 6 (batch 1621) loss 6.3160161881335535 valid acc 13/16 best valid acc 16
Epoch 6 (batch 1626) loss 3.654911639668315 valid acc 11/16 best valid acc 16
Epoch 6 (batch 1631) loss 3.0656593566970716 valid acc 13/16 best valid acc 16
Epoch 6 (batch 1636) loss 2.5767209192464993 valid acc 13/16 best valid acc 16
Epoch 6 (batch 1641) loss 2.5790004977583867 valid acc 14/16 best valid acc 16
Epoch 6 (batch 1646) loss 3.602982933123279 valid acc 13/16 best valid acc 16
Epoch 6 (batch 1651) loss 2.7694434007450752 valid acc 13/16 best valid acc 16
Epoch 6 (batch 1656) loss 1.5733333013704804 valid acc 12/16 best valid acc 16
Epoch 6 (batch 1661) loss 1.1292772123455723 valid acc 14/16 best valid acc 16
Epoch 6 (batch 1666) loss 2.0293697733810836 valid acc 10/16 best valid acc 16
Epoch 6 (batch 1671) loss 2.2525747152890583 valid acc 11/16 best valid acc 16
Epoch 6 (batch 1676) loss 4.568271708308632 valid acc 13/16 best valid acc 16
Epoch 6 (batch 1681) loss 3.0775770924962886 valid acc 13/16 best valid acc 16
Epoch 6 (batch 1686) loss 2.5050520200448534 valid acc 12/16 best valid acc 16
Epoch 6 (batch 1691) loss 2.7068658085968695 valid acc 14/16 best valid acc 16
Epoch 6 (batch 1696) loss 1.9481038362678635 valid acc 13/16 best valid acc 16
Epoch 6 (batch 1701) loss 1.7475513805075875 valid acc 12/16 best valid acc 16
Epoch 6 (batch 1706) loss 1.0008436852432372 valid acc 12/16 best valid acc 16
Epoch 6 (batch 1711) loss 2.49374836311767 valid acc 13/16 best valid acc 16
Epoch 6 (batch 1716) loss 3.8685724761676075 valid acc 12/16 best valid acc 16
Epoch 6 (batch 1721) loss 2.2472741982605564 valid acc 13/16 best valid acc 16
Epoch 6 (batch 1726) loss 2.337163720541705 valid acc 12/16 best valid acc 16
Epoch 6 (batch 1731) loss 7.2275620157097995 valid acc 11/16 best valid acc 16
Epoch 6 (batch 1736) loss 4.441349030351376 valid acc 13/16 best valid acc 16
Epoch 6 (batch 1741) loss 8.48893016582852 valid acc 9/16 best valid acc 16
Epoch 6 (batch 1746) loss 4.462939363222866 valid acc 11/16 best valid acc 16
Epoch 6 (batch 1751) loss 10.129930003134266 valid acc 9/16 best valid acc 16
Epoch 6 (batch 1756) loss 16.169949159750914 valid acc 10/16 best valid acc 16
Epoch 6 (batch 1761) loss 8.411678908516254 valid acc 8/16 best valid acc 16
Epoch 6 (batch 1766) loss 6.185758201382161 valid acc 10/16 best valid acc 16
Epoch 6 (batch 1771) loss 9.128718149688105 valid acc 6/16 best valid acc 16
Epoch 6 (batch 1776) loss 8.083336767474739 valid acc 10/16 best valid acc 16
Epoch 6 (batch 1781) loss 6.462187006807768 valid acc 10/16 best valid acc 16
Epoch 6 (batch 1786) loss 4.463080297227811 valid acc 11/16 best valid acc 16
Epoch 6 (batch 1791) loss 6.691746676520842 valid acc 11/16 best valid acc 16
Epoch 6 (batch 1796) loss 7.514539582021724 valid acc 8/16 best valid acc 16
Epoch 6 (batch 1801) loss 6.004726677472271 valid acc 12/16 best valid acc 16
Epoch 6 (batch 1806) loss 3.908426988530144 valid acc 11/16 best valid acc 16
Epoch 6 (batch 1811) loss 6.734630635114673 valid acc 11/16 best valid acc 16
Epoch 6 (batch 1816) loss 6.562559993778347 valid acc 11/16 best valid acc 16
Epoch 6 (batch 1821) loss 6.053916855945957 valid acc 11/16 best valid acc 16
Epoch 6 (batch 1826) loss 7.1840246569722765 valid acc 10/16 best valid acc 16
Epoch 6 (batch 1831) loss 6.531641961770557 valid acc 11/16 best valid acc 16
Epoch 6 (batch 1836) loss 5.388215380379006 valid acc 13/16 best valid acc 16
Epoch 6 (batch 1841) loss 6.728104199667162 valid acc 12/16 best valid acc 16
Epoch 6 (batch 1846) loss 5.767253044633819 valid acc 12/16 best valid acc 16
Epoch 6 (batch 1851) loss 5.4141765081765625 valid acc 14/16 best valid acc 16
Epoch 6 (batch 1856) loss 7.48490736147887 valid acc 13/16 best valid acc 16
Epoch 6 (batch 1861) loss 4.620030307363576 valid acc 12/16 best valid acc 16
Epoch 6 (batch 1866) loss 4.340047243521239 valid acc 11/16 best valid acc 16
Epoch 6 (batch 1871) loss 3.2579754997551436 valid acc 10/16 best valid acc 16
Epoch 7 (batch 1873) loss 0.6373399918253329 valid acc 7/16 best valid acc 16
Epoch 7 (batch 1878) loss 5.87220896247762 valid acc 11/16 best valid acc 16
Epoch 7 (batch 1883) loss 4.700823734878298 valid acc 12/16 best valid acc 16
Epoch 7 (batch 1888) loss 3.3751296786631615 valid acc 13/16 best valid acc 16
Epoch 7 (batch 1893) loss 3.0496809063743826 valid acc 12/16 best valid acc 16
Epoch 7 (batch 1898) loss 3.8221309906197054 valid acc 11/16 best valid acc 16
Epoch 7 (batch 1903) loss 7.056803938620838 valid acc 11/16 best valid acc 16
Epoch 7 (batch 1908) loss 4.532098925808915 valid acc 11/16 best valid acc 16
Epoch 7 (batch 1913) loss 4.207116989922939 valid acc 9/16 best valid acc 16
Epoch 7 (batch 1918) loss 4.017246304842528 valid acc 10/16 best valid acc 16
Epoch 7 (batch 1923) loss 3.4914574904239517 valid acc 13/16 best valid acc 16
Epoch 7 (batch 1928) loss 5.9325209911286825 valid acc 12/16 best valid acc 16
Epoch 7 (batch 1933) loss 6.78310545624357 valid acc 12/16 best valid acc 16
Epoch 7 (batch 1938) loss 7.140694791322202 valid acc 12/16 best valid acc 16
Epoch 7 (batch 1943) loss 5.5937135362018635 valid acc 11/16 best valid acc 16
Epoch 7 (batch 1948) loss 4.217423242473127 valid acc 12/16 best valid acc 16
Epoch 7 (batch 1953) loss 3.855446096145813 valid acc 12/16 best valid acc 16
Epoch 7 (batch 1958) loss 4.5168723902641315 valid acc 12/16 best valid acc 16
Epoch 7 (batch 1963) loss 3.971122552875755 valid acc 10/16 best valid acc 16
Epoch 7 (batch 1968) loss 4.205689337564028 valid acc 14/16 best valid acc 16
Epoch 7 (batch 1973) loss 4.889804342771416 valid acc 9/16 best valid acc 16
Epoch 7 (batch 1978) loss 3.9732638868493995 valid acc 12/16 best valid acc 16
Epoch 7 (batch 1983) loss 4.191654555419666 valid acc 13/16 best valid acc 16
Epoch 7 (batch 1988) loss 5.078615578642442 valid acc 13/16 best valid acc 16
Epoch 7 (batch 1993) loss 3.904521202285718 valid acc 12/16 best valid acc 16
Epoch 7 (batch 1998) loss 3.7226652672237623 valid acc 13/16 best valid acc 16
Epoch 7 (batch 2003) loss 4.964175431942074 valid acc 13/16 best valid acc 16
Epoch 7 (batch 2008) loss 2.9750573245906415 valid acc 14/16 best valid acc 16
Epoch 7 (batch 2013) loss 4.257052873671762 valid acc 12/16 best valid acc 16
Epoch 7 (batch 2018) loss 2.5593086558107006 valid acc 11/16 best valid acc 16
Epoch 7 (batch 2023) loss 6.516584069281771 valid acc 12/16 best valid acc 16
Epoch 7 (batch 2028) loss 7.577832148851926 valid acc 11/16 best valid acc 16
Epoch 7 (batch 2033) loss 4.790932196478007 valid acc 11/16 best valid acc 16
Epoch 7 (batch 2038) loss 7.239724831371839 valid acc 12/16 best valid acc 16
Epoch 7 (batch 2043) loss 6.6205541079159 valid acc 13/16 best valid acc 16
Epoch 7 (batch 2048) loss 7.686804134196064 valid acc 14/16 best valid acc 16
Epoch 7 (batch 2053) loss 7.058530962192526 valid acc 10/16 best valid acc 16
Epoch 7 (batch 2058) loss 4.225553799926575 valid acc 11/16 best valid acc 16
Epoch 7 (batch 2063) loss 4.8919763186748595 valid acc 12/16 best valid acc 16
Epoch 7 (batch 2068) loss 5.4587687484148235 valid acc 11/16 best valid acc 16
Epoch 7 (batch 2073) loss 7.983186971274641 valid acc 12/16 best valid acc 16
Epoch 7 (batch 2078) loss 4.87012493485486 valid acc 12/16 best valid acc 16
Epoch 7 (batch 2083) loss 3.2478817647665825 valid acc 10/16 best valid acc 16
Epoch 7 (batch 2088) loss 6.654778056867368 valid acc 12/16 best valid acc 16
Epoch 7 (batch 2093) loss 5.610581714221267 valid acc 10/16 best valid acc 16
Epoch 7 (batch 2098) loss 3.5169781371871607 valid acc 12/16 best valid acc 16
Epoch 7 (batch 2103) loss 4.529389191081304 valid acc 11/16 best valid acc 16
Epoch 7 (batch 2108) loss 4.196442536224946 valid acc 11/16 best valid acc 16
Epoch 7 (batch 2113) loss 3.488684391030273 valid acc 14/16 best valid acc 16
Epoch 7 (batch 2118) loss 2.7948410236891608 valid acc 11/16 best valid acc 16
Epoch 7 (batch 2123) loss 4.0051677626062645 valid acc 12/16 best valid acc 16
Epoch 7 (batch 2128) loss 3.861071674617148 valid acc 14/16 best valid acc 16
Epoch 7 (batch 2133) loss 4.431295354656341 valid acc 14/16 best valid acc 16
Epoch 7 (batch 2138) loss 1.964955094514621 valid acc 13/16 best valid acc 16
Epoch 7 (batch 2143) loss 4.155503935430142 valid acc 15/16 best valid acc 16
Epoch 7 (batch 2148) loss 1.874396502102214 valid acc 14/16 best valid acc 16
Epoch 7 (batch 2153) loss 5.706455563204595 valid acc 14/16 best valid acc 16
Epoch 7 (batch 2158) loss 4.043621280135856 valid acc 13/16 best valid acc 16
Epoch 7 (batch 2163) loss 3.2461178001406594 valid acc 14/16 best valid acc 16
Epoch 7 (batch 2168) loss 4.460339637439916 valid acc 12/16 best valid acc 16
Epoch 7 (batch 2173) loss 4.444852613804194 valid acc 14/16 best valid acc 16
Epoch 7 (batch 2178) loss 3.478282039973405 valid acc 11/16 best valid acc 16
Epoch 7 (batch 2183) loss 3.411545442571085 valid acc 14/16 best valid acc 16
Epoch 8 (batch 2185) loss 0.3047028246243165 valid acc 12/16 best valid acc 16
Epoch 8 (batch 2190) loss 4.112277384221719 valid acc 14/16 best valid acc 16
Epoch 8 (batch 2195) loss 3.648733510568036 valid acc 13/16 best valid acc 16
Epoch 8 (batch 2200) loss 4.596255545925054 valid acc 12/16 best valid acc 16
Epoch 8 (batch 2205) loss 2.5515736660975135 valid acc 12/16 best valid acc 16
Epoch 8 (batch 2210) loss 2.8759675562997637 valid acc 12/16 best valid acc 16
Epoch 8 (batch 2215) loss 5.465977002039265 valid acc 9/16 best valid acc 16
Epoch 8 (batch 2220) loss 8.519859451350808 valid acc 11/16 best valid acc 16
Epoch 8 (batch 2225) loss 11.365581600226905 valid acc 10/16 best valid acc 16
Epoch 8 (batch 2230) loss 8.611902113532441 valid acc 7/16 best valid acc 16
Epoch 8 (batch 2235) loss 6.447169831909636 valid acc 9/16 best valid acc 16
Epoch 8 (batch 2240) loss 13.46420673683146 valid acc 9/16 best valid acc 16
Epoch 8 (batch 2245) loss 13.832679570510015 valid acc 7/16 best valid acc 16
Epoch 8 (batch 2250) loss 11.772715323271697 valid acc 9/16 best valid acc 16
Epoch 8 (batch 2255) loss 8.811512682472179 valid acc 8/16 best valid acc 16
Epoch 8 (batch 2260) loss 6.734515582147015 valid acc 9/16 best valid acc 16
Epoch 8 (batch 2265) loss 8.163795414711696 valid acc 11/16 best valid acc 16
Epoch 8 (batch 2270) loss 9.092278569594784 valid acc 9/16 best valid acc 16
Epoch 8 (batch 2275) loss 6.854439010313185 valid acc 9/16 best valid acc 16
Epoch 8 (batch 2280) loss 8.226577120644471 valid acc 8/16 best valid acc 16
Epoch 8 (batch 2285) loss 6.971434387740708 valid acc 6/16 best valid acc 16
Epoch 8 (batch 2290) loss 6.300381481627154 valid acc 9/16 best valid acc 16
Epoch 8 (batch 2295) loss 4.978934454942297 valid acc 12/16 best valid acc 16
Epoch 8 (batch 2300) loss 6.3982400941904665 valid acc 10/16 best valid acc 16
Epoch 8 (batch 2305) loss 4.225227401176036 valid acc 11/16 best valid acc 16
Epoch 8 (batch 2310) loss 6.619435552513866 valid acc 12/16 best valid acc 16
Epoch 8 (batch 2315) loss 6.552601411905181 valid acc 12/16 best valid acc 16
Epoch 8 (batch 2320) loss 4.332018886627318 valid acc 11/16 best valid acc 16
Epoch 8 (batch 2325) loss 6.7956178899984465 valid acc 11/16 best valid acc 16
Epoch 8 (batch 2330) loss 4.3536887936608295 valid acc 12/16 best valid acc 16
Epoch 8 (batch 2335) loss 7.2009209674470664 valid acc 12/16 best valid acc 16
Epoch 8 (batch 2340) loss 7.627091000071504 valid acc 10/16 best valid acc 16
Epoch 8 (batch 2345) loss 7.222885991095469 valid acc 9/16 best valid acc 16
Epoch 8 (batch 2350) loss 4.968438072334583 valid acc 7/16 best valid acc 16
Epoch 8 (batch 2355) loss 9.050691058389383 valid acc 8/16 best valid acc 16
Epoch 8 (batch 2360) loss 5.529815813586932 valid acc 9/16 best valid acc 16
Epoch 8 (batch 2365) loss 7.338984352385013 valid acc 8/16 best valid acc 16
...


SENTIMENT_TRAINING:
Epoch 1, loss 31.488784470278063, train accuracy: 52.00%
Validation accuracy: 52.00%
Best Valid accuracy: 52.00%
Epoch 2, loss 31.29225155276331, train accuracy: 52.89%
Validation accuracy: 52.00%
Best Valid accuracy: 52.00%
Epoch 3, loss 31.175102750033595, train accuracy: 49.33%
Validation accuracy: 49.00%
Best Valid accuracy: 52.00%
Epoch 4, loss 30.994123955416985, train accuracy: 54.67%
Validation accuracy: 56.00%
Best Valid accuracy: 56.00%
Epoch 5, loss 30.73291337813986, train accuracy: 55.11%
Validation accuracy: 60.00%
Best Valid accuracy: 60.00%
Epoch 6, loss 30.647442796392284, train accuracy: 60.44%
Validation accuracy: 65.00%
Best Valid accuracy: 65.00%
Epoch 7, loss 30.433215164007546, train accuracy: 60.00%
Validation accuracy: 58.00%
Best Valid accuracy: 65.00%
Epoch 8, loss 30.16247381938178, train accuracy: 59.11%
Validation accuracy: 63.00%
Best Valid accuracy: 65.00%
Epoch 9, loss 29.856989638579005, train accuracy: 62.44%
Validation accuracy: 61.00%
Best Valid accuracy: 65.00%
Epoch 10, loss 29.454456368729495, train accuracy: 64.89%
Validation accuracy: 69.00%
Best Valid accuracy: 69.00%
Epoch 11, loss 29.132865083759206, train accuracy: 66.22%
Validation accuracy: 64.00%
Best Valid accuracy: 69.00%
Epoch 12, loss 28.75335479565784, train accuracy: 67.33%
Validation accuracy: 63.00%
Best Valid accuracy: 69.00%
Epoch 13, loss 28.42142195923316, train accuracy: 68.89%
Validation accuracy: 71.00%
Best Valid accuracy: 71.00%
Epoch 14, loss 27.678610860310343, train accuracy: 70.44%
Validation accuracy: 61.00%
Best Valid accuracy: 71.00%
Epoch 15, loss 27.470710049365795, train accuracy: 70.67%
Validation accuracy: 70.00%
Best Valid accuracy: 71.00%
Epoch 16, loss 26.983163634694513, train accuracy: 71.11%
Validation accuracy: 70.00%
Best Valid accuracy: 71.00%
Epoch 17, loss 26.01561116141862, train accuracy: 73.11%
Validation accuracy: 71.00%
Best Valid accuracy: 71.00%
Epoch 18, loss 25.487947452465754, train accuracy: 70.89%
Validation accuracy: 71.00%
Best Valid accuracy: 71.00%
Epoch 19, loss 25.0324407801461, train accuracy: 71.33%
Validation accuracy: 69.00%
Best Valid accuracy: 71.00%
Epoch 20, loss 24.243336066940575, train accuracy: 75.33%
Validation accuracy: 75.00%
Best Valid accuracy: 75.00%
Epoch 21, loss 23.89169595990919, train accuracy: 76.22%
Validation accuracy: 78.00%
Best Valid accuracy: 78.00%
Epoch 22, loss 23.27745096783514, train accuracy: 78.44%
Validation accuracy: 71.00%
Best Valid accuracy: 78.00%
Epoch 23, loss 22.480985161578605, train accuracy: 76.67%
Validation accuracy: 72.00%
Best Valid accuracy: 78.00%
