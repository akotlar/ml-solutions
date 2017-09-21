use 5.10.0;
use strict;
use warnings;
use DDP;
use MCE;
use MCE::Loop;

if (@ARGV < 3) {
  die "need trainLabels trainingFeatures testLabels testFeatures";
}

(open my $trainLabelsFh, "<", $ARGV[0]) or die "Couldn't open $ARGV[0] for reading";
(open my $trainFeatFh, "<", $ARGV[1]) or die "Couldn't open $ARGV[1] for reading";
(open my $testLabelsFh, "<", $ARGV[2]) or die "Couldn't open $ARGV[0] for reading";
(open my $testFeatFh, "<", $ARGV[3]) or die "Couldn't open $ARGV[2] for reading";

############ Each row in feature file encodes a non-0 entry ###################
#1st col is message number
#2nd is ID of the word
#3rd is the # of occurances (count)
#Ex:
#2 977 2 says email 2 has 2 occurances of word 977

# Tip: use logsumexp, avoid numerical underflow

# a) Goal: AUC (ROC AUC) and classification error on training set


my $totalNumberOfEmails;
my $spam;
my $ham;

#holds id => {
# spam => int, totalCount => int, totalSpam => int, totalHam => int,
# freqSpam => float, freqHam => float, pSpamGivenWord => float,  pHamGivenWord => float, 
#}

# groups by id
my %words;

# groups by spam or not
my @logLikelihoods;

#assuming 0 is ham 1 is spam
my @labels = <$trainLabelsFh>;

chomp @labels;

my $spamCount = 0;
my $hamCount = 0;

my %documents;

my $totalWordOccurances = 0;
while(<$trainFeatFh>) {
  chomp;

  my @fields = split /\s/, $_;

  # fields[0] == message number - 1 for index
  my $isSpam = $labels[$fields[0] - 1];
  my $id = $fields[1];
  my $occurs = $fields[2];

  $totalWordOccurances += $occurs;
  $words{$id} //= 1;

  if(!exists $documents{$fields[0]}) {
    if($isSpam) {
      $spamCount++
    } else {
      $hamCount++
    }

    $documents{$fields[0]} = {
      words => [$id],
      occurances => [$occurs],
      class => $isSpam,
    };
  } else {
    push @{$documents{$fields[0]}{words}}, $id;
    push @{$documents{$fields[0]}{occurances}}, $occurs;
  }

  if(!exists $words{$id}) {
    $words{$id} = 1;
    $logLikelihoods[$isSpam] = { $id => $occurs };

    next;
  }

  $logLikelihoods[$isSpam]{$id} += $occurs;
  
}

my $pSpam = $spamCount / ($spamCount + $hamCount);
my $pHam = 1 - $pSpam;

my @classPriors = ($pSpam, $pHam);
my @logClassPriors = (log($pSpam), log($pHam));

#let's calculate P(w|spam)
#let's calculate P(spam|word)

my $totalUniqueWords = keys %words;

my %likelihoodWords;
my @logConditional;

# use smoothing : + 1 to numerator + unique # words in all categories for denominator
# this is called laplace smoothing
# furthermore, for features never seen in training data, we have a singular value, likelihood 0:
my $unseenLogLikelihood = log(1) - log($totalWordOccurances + $totalUniqueWords);

for my $wordId (keys %words) {
  for my $i ( 0 .. 1) {
    my $likelihood = $logLikelihoods[$i]{$wordId} || 0;

    # smoothing factors 1 and totalUniqueWords
    # to protect against sparseness (not all words will appear in all categories)
    $logLikelihoods[$i]{$wordId} = log(($likelihood + 1) / ($totalWordOccurances + $totalUniqueWords));
    # my $test = log($likelihoods[$i]{$wordId}) + $logClassPriors[$i];
    # p $test;
    push @logConditional, $logLikelihoods[$i]{$wordId} + $logClassPriors[$i];
  }
}

my $maxLog = $logConditional[0];

for my $log (@logConditional) {
  if($maxLog < $log) {
    $maxLog = $log;
  }
}

sub printResults {
  my $correctRef = shift;

  return sub {
    my $isCorrect = shift;

    if($isCorrect) {
      $$correctRef++;
    }
  }
}

sub accumulate {
  my ($docId, $doc, $trainedLogClassPriors, $trainedLikelihoods) = @_;

  my @words = @{$doc->{words}};

  my $actualClass = $doc->{class};
  my @occurances = @{$doc->{occurances}};

  # say "Classifying $docId";

  my $bestPrediction;
  my $bestClass;

  my $logLikelihood;
  my $logClassPrior;
  my $occurs;
  my $prediction;
  for my $i ( 0 .. 1) {
    my $sum = 0;
    $logClassPrior = $trainedLogClassPriors->[$i];

    # say "log class prior for $i is $logClassPrior";
    my $idx = -1;
    for my $wordId (@words) {
      $idx++;

      $logLikelihood = $trainedLikelihoods->[$i]{$wordId};
      
      if(!defined $logLikelihood) {
        say STDERR "Couldn't find likelihood for doc $docId, word $wordId";
        $logLikelihood = $unseenLogLikelihood
      }

      $occurs = $occurances[$idx];
      
      # say "occurs $occurs time for word $wordId, doc $docId";
      
      
      for my $p ( 1 .. $occurs) {
        $sum += $logLikelihood;
      }
      
    }

    $prediction = $logClassPrior + $sum;

      if((!defined $bestPrediction || $bestPrediction < $prediction)) {
        $bestClass = $i;
        $bestPrediction = $prediction;
      }
  }

  return $bestClass, $actualClass;
}


say "\n--------------Classifying Training dataset----------------\n";

my $totalTrainingDocuments = scalar keys %documents;

my $correct = 0;
MCE::Loop::init {
   chunk_size => 1, max_workers => 3,
   gather => printResults(\$correct),
};

#classify the class
mce_loop {
  my ($mce, $chunk_ref, $chunk_id) = @_;

  my ($bestClass, $actualClass) = accumulate($_, $documents{$_}, \@logClassPriors, \@logLikelihoods);
    
  if($bestClass != $actualClass) {
    say "We think $_ is " . ($bestClass == 0 ? "Spam" : "Ham") . " and it really is " . $actualClass;
  }

  MCE->gather($bestClass == $actualClass);
} keys %documents;

say "\n\nCorrect count for training: $correct out of $totalTrainingDocuments : " . ($correct * 100 / $totalTrainingDocuments) . "%";

#assuming 0 is ham 1 is spam
my @testLabels = <$testLabelsFh>;
chomp @testLabels;

%documents = ();
while(<$testFeatFh>) {
  chomp;

  my @fields = split /\s/, $_;

  # fields[0] == message number - 1 for index
  my $isSpam = $testLabels[$fields[0] - 1];
  my $id = $fields[1];
  my $occurs = $fields[2];

  if(!exists $documents{$fields[0]}) {
    $documents{$fields[0]} = {
      words => [$id],
      occurances => [$occurs],
      class => $isSpam,
    };
  } else {
    push @{$documents{$fields[0]}{words}}, $id;
    push @{$documents{$fields[0]}{occurances}}, $occurs;
  }
}

my $totalTestDocuments = scalar keys %documents;

my $correctTest = 0;
MCE::Loop::init {
   chunk_size => 1, max_workers => 3,
   gather => printResults(\$correctTest),
};

say "\n--------------Classifying Test dataset----------------\n";

#classify the class
mce_loop {
  my ($mce, $chunk_ref, $chunk_id) = @_;

  my ($bestClass, $actualClass) = accumulate($_, $documents{$_}, \@logClassPriors, \@logLikelihoods);
    
  if($bestClass != $actualClass) {
    say "We think $_ is " . ($bestClass == 0 ? "Spam" : "Ham") . " and it really is " . $actualClass;
  }

  MCE->gather($bestClass == $actualClass);
} keys %documents;

say "\n\nCorrect count for test: $correctTest out of $totalTestDocuments : "  . ($correctTest * 100 / $totalTestDocuments) . "%";
