# Battery — top-3 inferences (merged 5-seed gibbs+bd posterior)

Channel align, P=256, gibbs+bd indel rejuv, 5 seeds (evidence-weighted merge).  87 items.

Per item: **observed** (model input) → **intended** (the gold restoration), the expected behavior, then the 3 highest-probability hypotheses.  Markers: `←intended` = matches the gold restoration, `←observed` = equals the input (a no-op/keep).

## SUBW-01a  ·  SUBW / implausible / expect-edit
- observed: `The storyteller told an amusing antidote.`
- intended: `The storyteller told an amusing anecdote.`
- logZ -70.02, seed-spread 1.3 nats

  1. `0.731`  The storyteller told an amusing anecdote.  ←intended
  2. `0.163`  It the storyteller told an amusing anecdote.
  3. `0.042`  The storyteller told an amusing antidote.  ←observed

## SUBW-01b  ·  SUBW / plausible / expect-keep
- observed: `The doctor prepared a fast-acting antidote.`
- intended: `The doctor prepared a fast-acting antidote.`
- logZ -53.16, seed-spread 7.064 nats

  1. `0.840`  The doctor prepared a fast-acting antidote.  ←intended
  2. `0.138`  The doctor prepares a fast-acting antidote.
  3. `0.012`  The Doctor prepared a fast-acting antidote.

## SUBW-02a  ·  SUBW / implausible / expect-edit
- observed: `The medic treated the wound to prevent an inflection.`
- intended: `The medic treated the wound to prevent an infection.`
- logZ -57.67, seed-spread 4.896 nats

  1. `0.668`  The medic treated the wound to prevent an infection.  ←intended
  2. `0.268`  The media treated the wound to prevent an infection.
  3. `0.048`  The medic created the wound to prevent an infection.

## SUBW-02b  ·  SUBW / plausible / expect-keep
- observed: `The teacher explained the verb's inflection.`
- intended: `The teacher explained the verb's inflection.`
- logZ -48.05, seed-spread 2.095 nats

  1. `0.980`  The teacher explained the verb's inflection.  ←intended
  2. `0.015`  The Teacher explained the verb's inflection.
  3. `0.002`  The teacher explained the verb's inflection

## SUBW-03a  ·  SUBW / implausible / expect-edit
- observed: `The judge cited an important president.`
- intended: `The judge cited an important precedent.`
- logZ -46.34, seed-spread 2.545 nats

  1. `0.616`  The judge cited an important president.  ←observed
  2. `0.183`  The judge cited an important President.
  3. `0.150`  The judge cited an important precedent.  ←intended

## SUBW-03b  ·  SUBW / plausible / expect-keep
- observed: `The board elected a new president.`
- intended: `The board elected a new president.`
- logZ -33.95, seed-spread 0.862 nats

  1. `0.755`  The Board elected a new president.
  2. `0.148`  The Board elected a new President.
  3. `0.080`  The board elected a new president.  ←intended

## SUBW-04a  ·  SUBW / implausible / expect-edit
- observed: `The explorers trekked across the scorching dessert.`
- intended: `The explorers trekked across the scorching desert.`
- logZ -65.78, seed-spread 11.166 nats

  1. `0.987`  The explorers trekked across the scorching desert.  ←intended
  2. `0.008`  The Explorer trekked across the scorching desert.
  3. `0.004`  The explorers trekked across the scorching dessert.  ←observed

## SUBW-04b  ·  SUBW / plausible / expect-keep
- observed: `She ordered a rich chocolate dessert.`
- intended: `She ordered a rich chocolate dessert.`
- logZ -43.74, seed-spread 1.864 nats

  1. `0.987`  She ordered a rich chocolate dessert.  ←intended
  2. `0.006`  She ordered a rich Chocolate dessert.
  3. `0.003`  She ordered a Rich chocolate dessert.

## SUBW-05a  ·  SUBW / implausible / expect-edit
- observed: `He wore a very causal shirt to the wedding.`
- intended: `He wore a very casual shirt to the wedding.`
- logZ -49.48, seed-spread 3.82 nats

  1. `0.996`  He wore a very casual shirt to the wedding.  ←intended
  2. `0.002`  He wore a casual shirt to the wedding.
  3. `0.001`  He wore a very casual shirt to the Wedding.

## SUBW-05b  ·  SUBW / plausible / expect-keep
- observed: `They studied the causal link between the variables.`
- intended: `They studied the causal link between the variables.`
- logZ -47.00, seed-spread 3.123 nats

  1. `0.895`  They studied the causal link between the variables.  ←intended
  2. `0.064`  They studied the causal link between the two variables.
  3. `0.031`  They studied the causal links between the variables.

## SUBW-06a  ·  SUBW / implausible / expect-edit
- observed: `She gave him a lovely complement on his cooking.`
- intended: `She gave him a lovely compliment on his cooking.`
- logZ -50.39, seed-spread 0.941 nats

  1. `0.989`  She gave him a lovely compliment on his cooking.  ←intended
  2. `0.004`  She gave him a lovely compliment in his cooking.
  3. `0.002`  She gave him a lovely compliment on his booking.

## SUBW-06b  ·  SUBW / plausible / expect-keep
- observed: `The wine was a perfect complement to the meal.`
- intended: `The wine was a perfect complement to the meal.`
- logZ -45.53, seed-spread 0.977 nats

  1. `0.922`  The wine was a perfect complement to the meal.  ←intended
  2. `0.023`  The win was a perfect complement to the meal.
  3. `0.023`  The wine was a perfect complement to the meat.

## SUBN-01a  ·  SUBN / implausible / expect-edit
- observed: `Did you recieve the message?`
- intended: `Did you receive the message?`
- logZ -29.78, seed-spread 1.294 nats

  1. `0.866`  Did you receive the message?  ←intended
  2. `0.116`  Did you recieve the message?  ←observed
  3. `0.012`  Did you receive the Message?

## SUBN-01b  ·  SUBN / plausible / expect-keep
- observed: `Did you receive the message?`
- intended: `Did you receive the message?`
- logZ -27.16, seed-spread 0.441 nats

  1. `0.969`  Did you receive the message?  ←intended
  2. `0.018`  Did you receive the Message?
  3. `0.011`  Did You receive the message?

## SUBN-02a  ·  SUBN / implausible / expect-edit
- observed: `The boy did an experimemt today.`
- intended: `The boy did an experiment today.`
- logZ -43.95, seed-spread 7.153 nats

  1. `0.826`  The boy did an experiment today.  ←intended
  2. `0.122`  The Boy did an experiment today.
  3. `0.015`  The boys did an experiment today.

## SUBN-02b  ·  SUBN / plausible / expect-keep
- observed: `The boy did an experiment today.`
- intended: `The boy did an experiment today.`
- logZ -42.81, seed-spread 3.301 nats

  1. `0.818`  The boy did an experiment today.  ←intended
  2. `0.161`  The Boy did an experiment today.
  3. `0.010`  The boy did an experiment.

## SUBN-03a  ·  SUBN / implausible / expect-edit
- observed: `She definately agreed with the plan.`
- intended: `She definitely agreed with the plan.`
- logZ -39.83, seed-spread 17.849 nats

  1. `0.613`  She definitely agreed with the plan.  ←intended
  2. `0.351`  She definately agreed with the plan.  ←observed
  3. `0.032`  She definately agreed with the Plan.

## SUBN-03b  ·  SUBN / plausible / expect-keep
- observed: `She definitely agreed with the plan.`
- intended: `She definitely agreed with the plan.`
- logZ -35.36, seed-spread 2.044 nats

  1. `0.968`  She definitely agreed with the plan.  ←intended
  2. `0.022`  She definitely agreed with the Plan.
  3. `0.008`  She definitely agrees with the plan.

## SUBN-04a  ·  SUBN / implausible / expect-edit
- observed: `The mountian was covered in snow.`
- intended: `The mountain was covered in snow.`
- logZ -37.08, seed-spread 15.483 nats

  1. `0.991`  The mountain was covered in snow.  ←intended
  2. `0.004`  The mountain was covered in snow
  3. `0.002`  The mountain was covered in the snow.

## SUBN-04b  ·  SUBN / plausible / expect-keep
- observed: `The mountain was covered in snow.`
- intended: `The mountain was covered in snow.`
- logZ -34.58, seed-spread 1.364 nats

  1. `0.987`  The mountain was covered in snow.  ←intended
  2. `0.007`  The Mountain was covered in snow.
  3. `0.002`  The mountain was covered in snow

## SUBN-05a  ·  SUBN / implausible / expect-edit
- observed: `They reciesofted the parcel yesterday.`
- intended: `They received the parcel yesterday.`
- logZ -60.91, seed-spread 8.067 nats

  1. `0.889`  They the parcel yesterday.
  2. `0.086`  They have the parcel yesterday.
  3. `0.008`  They were the parcel yesterday.

## SUBN-06a  ·  SUBN / implausible / expect-edit
- observed: `The chef seasoned the suop.`
- intended: `The chef seasoned the soup.`
- logZ -51.31, seed-spread 12.348 nats

  1. `0.593`  The chef seasoned the soup.  ←intended
  2. `0.170`  The chef seasoned the shop.
  3. `0.111`  The Chef seasoned the soup.

## SUBN-06b  ·  SUBN / plausible / expect-keep
- observed: `The chef seasoned the soup.`
- intended: `The chef seasoned the soup.`
- logZ -45.59, seed-spread 10.238 nats

  1. `0.777`  The chef seasoned the soup.  ←intended
  2. `0.081`  The chef seasoned the soul.
  3. `0.080`  The Chef seasoned the soup.

## DELTO-01a  ·  DEL_TO / implausible / expect-edit
- observed: `The mother gave the candle a daughter.`
- intended: `The mother gave the candle to a daughter.`
- logZ -47.27, seed-spread 1.209 nats

  1. `0.948`  The mother gave the candle a daughter.  ←observed
  2. `0.023`  The mother gave the candle to a daughter.  ←intended
  3. `0.019`  The Mother gave the candle a daughter.

## DELTO-01b  ·  DEL_TO / plausible / expect-keep
- observed: `The mother gave the daughter a candle.`
- intended: `The mother gave the daughter a candle.`
- logZ -40.69, seed-spread 3.605 nats

  1. `0.995`  The mother gave the daughter a candle.  ←intended
  2. `0.002`  The mother gave the daughter a handle.
  3. `0.002`  The Mother gave the daughter a candle.

## DELTO-02a  ·  DEL_TO / implausible / expect-edit
- observed: `The waiter served the soup the customers.`
- intended: `The waiter served the soup to the customers.`
- logZ -50.63, seed-spread 8.902 nats

  1. `0.397`  The waiter served the soup to the customers.  ←intended
  2. `0.167`  The waiter served the soup and the customers.
  3. `0.131`  The waiter served the soup the customers.  ←observed

## DELTO-02b  ·  DEL_TO / plausible / expect-keep
- observed: `The waiter served the customers the soup.`
- intended: `The waiter served the customers the soup.`
- logZ -46.48, seed-spread 5.043 nats

  1. `0.837`  The waiter served the customers the soup.  ←intended
  2. `0.103`  The waiter served the customer the soup.
  3. `0.015`  The waiter served the customers with the soup.

## DELTO-03a  ·  DEL_TO / implausible / expect-edit
- observed: `The teacher read the story the children.`
- intended: `The teacher read the story to the children.`
- logZ -46.50, seed-spread 4.415 nats

  1. `0.255`  The teacher read the story to the children.  ←intended
  2. `0.245`  The teacher read the story the children.  ←observed
  3. `0.207`  The teacher read the story of the children.

## DELTO-03b  ·  DEL_TO / plausible / expect-keep
- observed: `The teacher read the children the story.`
- intended: `The teacher read the children the story.`
- logZ -45.94, seed-spread 4.137 nats

  1. `0.899`  The teacher read the children the story.  ←intended
  2. `0.021`  The teacher read to the children the story.
  3. `0.016`  The teacher read the children story.

## DELTO-04a  ·  DEL_TO / implausible / expect-edit
- observed: `The judge awarded the trophy the winner.`
- intended: `The judge awarded the trophy to the winner.`
- logZ -45.63, seed-spread 2.678 nats

  1. `0.304`  The Judge awarded the trophy to the winner.
  2. `0.245`  The Judge awarded the trophy the winner.
  3. `0.173`  The judge awarded the trophy the winner.  ←observed

## DELTO-04b  ·  DEL_TO / plausible / expect-keep
- observed: `The judge awarded the winner the trophy.`
- intended: `The judge awarded the winner the trophy.`
- logZ -42.54, seed-spread 0.926 nats

  1. `0.545`  The Judge awarded the winner the trophy.
  2. `0.390`  The judge awarded the winner the trophy.  ←intended
  3. `0.038`  The judges awarded the winner the trophy.

## DELFOR-01a  ·  DEL_FOR / implausible / expect-edit
- observed: `The tailor sewed the dress the bride.`
- intended: `The tailor sewed the dress for the bride.`
- logZ -62.23, seed-spread 12.832 nats

  1. `0.580`  The tailor sewed the dress the bride.  ←observed
  2. `0.143`  The tailor sewed the dress for the bride.  ←intended
  3. `0.094`  The tailor sewed the dress to the bride.

## DELFOR-01b  ·  DEL_FOR / plausible / expect-keep
- observed: `The tailor sewed the bride the dress.`
- intended: `The tailor sewed the bride the dress.`
- logZ -57.71, seed-spread 15.78 nats

  1. `0.596`  The tailor sewed the bride the dress.  ←intended
  2. `0.106`  The tailor sewed the bride for the dress.
  3. `0.086`  The tailor sewed the bride on the dress.

## DELFOR-02a  ·  DEL_FOR / implausible / expect-edit
- observed: `The father cooked the dinner the family.`
- intended: `The father cooked the dinner for the family.`
- logZ -48.51, seed-spread 3.608 nats

  1. `0.409`  The father cooked the dinner for the family.  ←intended
  2. `0.228`  The father cooked the dinner the family.  ←observed
  3. `0.061`  The father cooked the dinner to the family.

## DELFOR-02b  ·  DEL_FOR / plausible / expect-keep
- observed: `The father cooked the family the dinner.`
- intended: `The father cooked the family the dinner.`
- logZ -48.46, seed-spread 5.195 nats

  1. `0.610`  The father cooked the family the dinner.  ←intended
  2. `0.282`  The father cooked the family dinner.
  3. `0.047`  The Father cooked the family the dinner.

## DELFOR-03a  ·  DEL_FOR / implausible / expect-edit
- observed: `The baker iced the cake the children.`
- intended: `The baker iced the cake for the children.`
- logZ -56.52, seed-spread 13.769 nats

  1. `0.362`  The baker iced the cake the children.  ←observed
  2. `0.172`  The Baker iced the cake the children.
  3. `0.090`  The Baker iced the cake for the children.

## DELFOR-03b  ·  DEL_FOR / plausible / expect-keep
- observed: `The baker iced the children the cake.`
- intended: `The baker iced the children the cake.`
- logZ -52.42, seed-spread 14.275 nats

  1. `0.823`  The baker iced the children the cake.  ←intended
  2. `0.089`  The Maker iced the children the cake.
  3. `0.015`  The Maker diced the children the cake.

## DELFROM-01a  ·  DEL_FROM / implausible / expect-edit
- observed: `The businessman benefited the tax law.`
- intended: `The businessman benefited from the tax law.`
- logZ -46.29, seed-spread 20.866 nats

  1. `0.711`  The businessman benefited from the tax law.  ←intended
  2. `0.156`  The businessman benefited from the Tax Law.
  3. `0.055`  The businessman benefited the tax law.  ←observed

## DELFROM-01b  ·  DEL_FROM / plausible / expect-keep
- observed: `The businessman benefited from the tax law.`
- intended: `The businessman benefited from the tax law.`
- logZ -47.83, seed-spread 11.315 nats

  1. `0.477`  The businessman benefited from the Tax Law.
  2. `0.465`  The businessman benefited from the tax law.  ←intended
  3. `0.039`  The businessman has benefited from the tax law.

## DELFROM-02a  ·  DEL_FROM / implausible / expect-edit
- observed: `The patient slowly recovered the illness.`
- intended: `The patient slowly recovered from the illness.`
- logZ -40.72, seed-spread 13.694 nats

  1. `0.582`  The patient slowly recovered the illness.  ←observed
  2. `0.407`  The patient slowly recovered from the illness.  ←intended
  3. `0.003`  The patient slowly recovered the illness

## DELFROM-02b  ·  DEL_FROM / plausible / expect-keep
- observed: `The patient slowly recovered from the illness.`
- intended: `The patient slowly recovered from the illness.`
- logZ -38.84, seed-spread 12.089 nats

  1. `1.000`  The patient slowly recovered from the illness.  ←intended
  2. `0.000`  The patient recovery slowly recovered from the illness.
  3. `0.000`  The patient has slowly recovered from the illness.

## INS-01a  ·  INS_DUP / implausible / expect-edit
- observed: `The boy handed handed the pencil to the girl.`
- intended: `The boy handed the pencil to the girl.`
- logZ -52.19, seed-spread 4.286 nats

  1. `0.970`  The boy handed handed the pencil to the girl.  ←observed
  2. `0.007`  The body handed handed the pencil to the girl.
  3. `0.007`  The boy handed the pencil to the girl.  ←intended

## INS-01b  ·  INS_DUP / plausible / expect-keep
- observed: `The boy handed the pencil to the girl.`
- intended: `The boy handed the pencil to the girl.`
- logZ -44.67, seed-spread 13.107 nats

  1. `0.966`  The boy handed the pencil to the girl.  ←intended
  2. `0.025`  The Boy handed the pencil to the girl.
  3. `0.004`  The boy handed the pencil to the Girl.

## INS-02a  ·  INS_DUP / implausible / expect-edit
- observed: `The cat sat on on the mat.`
- intended: `The cat sat on the mat.`
- logZ -45.60, seed-spread 4.456 nats

  1. `0.613`  The cat sat on the mat.  ←intended
  2. `0.184`  The cat sat on on the mat.  ←observed
  3. `0.091`  The cap sat on on the mat.

## INS-02b  ·  INS_DUP / plausible / expect-keep
- observed: `The cat sat on the mat.`
- intended: `The cat sat on the mat.`
- logZ -38.10, seed-spread 5.394 nats

  1. `0.971`  The cat sat on the mat.  ←intended
  2. `0.016`  The cap sat on the mat.
  3. `0.012`  The car sat on the mat.

## INS-03a  ·  INS_DUP / implausible / expect-edit
- observed: `She quickly quickly finished her lunch.`
- intended: `She quickly finished her lunch.`
- logZ -45.34, seed-spread 3.775 nats

  1. `0.853`  She quickly quickly finished her lunch.  ←observed
  2. `0.075`  She quickly and quickly finished her lunch.
  3. `0.066`  She quickly finished her lunch.  ←intended

## INS-03b  ·  INS_DUP / plausible / expect-keep
- observed: `She quickly finished her lunch.`
- intended: `She quickly finished her lunch.`
- logZ -37.40, seed-spread 1.515 nats

  1. `0.992`  She quickly finished her lunch.  ←intended
  2. `0.005`  She quickly finished her lunch
  3. `0.001`  She quickly finishes her lunch.

## CTRL-01  ·  CTRL / control / expect-keep
- observed: `The pirate buried the treasure.`
- intended: `The pirate buried the treasure.`
- logZ -36.40, seed-spread 17.741 nats

  1. `0.964`  The pirate buried the treasure.  ←intended
  2. `0.028`  The pirates buried the treasure.
  3. `0.004`  The pirate buried the Treasure.

## CTRL-02  ·  CTRL / control / expect-keep
- observed: `The chef cooked the salmon.`
- intended: `The chef cooked the salmon.`
- logZ -44.35, seed-spread 12.135 nats

  1. `0.989`  The chef cooked the salmon.  ←intended
  2. `0.011`  The Chef cooked the salmon.
  3. `0.000`  The chef who cooked the salmon.

## CTRL-03  ·  CTRL / control / expect-keep
- observed: `The astronomer photographed the comet.`
- intended: `The astronomer photographed the comet.`
- logZ -40.93, seed-spread 13.11 nats

  1. `0.970`  The astronomer photographed the comet.  ←intended
  2. `0.029`  The astronomers photographed the comet.
  3. `0.001`  The astronomer photographed with the comet.

## CTRL-04  ·  CTRL / control / expect-keep
- observed: `The gardener planted the tulips.`
- intended: `The gardener planted the tulips.`
- logZ -46.85, seed-spread 13.426 nats

  1. `0.893`  The gardener planted the tulips.  ←intended
  2. `0.059`  The gardener planted the tulip.
  3. `0.036`  The gardeners planted the tulips.

## LADDER-give-1  ·  LADDER / ladder / expect-edit
- observed: `The volunteer gave the blanket the children.`
- intended: `The volunteer gave the blanket to the children.`
- logZ -52.07, seed-spread 2.243 nats

  1. `0.592`  The volunteer gave the blanket the children.  ←observed
  2. `0.304`  The volunteer gave the blanket to the children.  ←intended
  3. `0.030`  The volunteers gave the blanket the children.

## LADDER-give-2  ·  LADDER / ladder / expect-edit
- observed: `The volunteer gave the shelter the children.`
- intended: `The volunteer gave the shelter to the children.`
- logZ -48.37, seed-spread 4.75 nats

  1. `0.863`  The volunteer gave the shelter the children.  ←observed
  2. `0.042`  The volunteer gave the shelter to the children.  ←intended
  3. `0.039`  The volunteer gave the shelter and the children.

## LADDER-give-3  ·  LADDER / ladder / expect-keep
- observed: `The volunteer gave the family the children.`
- intended: `The volunteer gave the family the children.`
- logZ -46.05, seed-spread 16.659 nats

  1. `0.967`  The volunteer gave the family the children.  ←intended
  2. `0.011`  The volunteer gave the family and the children.
  3. `0.008`  The volunteer gave the family with the children.

## LADDER-send-1  ·  LADDER / ladder / expect-edit
- observed: `The clerk sent the package the manager.`
- intended: `The clerk sent the package to the manager.`
- logZ -47.91, seed-spread 2.237 nats

  1. `0.694`  The clerk sent the package to the manager.  ←intended
  2. `0.131`  The clerk sent the package the manager.  ←observed
  3. `0.064`  The Clerk sent the package to the manager.

## LADDER-send-2  ·  LADDER / ladder / expect-edit
- observed: `The clerk sent the branch the manager.`
- intended: `The clerk sent the branch to the manager.`
- logZ -49.64, seed-spread 2.963 nats

  1. `0.297`  The clerk sent the branch the manager.  ←observed
  2. `0.196`  The Clerk sent the branch the manager.
  3. `0.119`  The clerk sent the branch to the manager.  ←intended

## LADDER-send-3  ·  LADDER / ladder / expect-keep
- observed: `The clerk sent the team the manager.`
- intended: `The clerk sent the team the manager.`
- logZ -47.03, seed-spread 1.593 nats

  1. `0.654`  The clerk sent the team the manager.  ←intended
  2. `0.120`  The clerk sent the team to the manager.
  3. `0.068`  The Clerk sent the team the manager.

## INS-04a  ·  INS_DUP / implausible / expect-edit
- observed: `She went to to the store.`
- intended: `She went to the store.`
- logZ -39.77, seed-spread 1.043 nats

  1. `0.717`  She went to the store.  ←intended
  2. `0.255`  She went to to the store.  ←observed
  3. `0.008`  She went to the Store.

## INS-04b  ·  INS_DUP / plausible / expect-keep
- observed: `She went to the store.`
- intended: `She went to the store.`
- logZ -31.83, seed-spread 0.807 nats

  1. `0.970`  She went to the store.  ←intended
  2. `0.027`  She went to the Store.
  3. `0.001`  She went to the stores.

## INS-05a  ·  INS_DUP / implausible / expect-edit
- observed: `He is is very tall.`
- intended: `He is very tall.`
- logZ -35.37, seed-spread 0.957 nats

  1. `0.525`  He is very tall.  ←intended
  2. `0.465`  He is is very tall.  ←observed
  3. `0.004`  He is is very talk.

## INS-05b  ·  INS_DUP / plausible / expect-keep
- observed: `He is very tall.`
- intended: `He is very tall.`
- logZ -28.33, seed-spread 0.552 nats

  1. `0.997`  He is very tall.  ←intended
  2. `0.002`  He is Very tall.
  3. `0.001`  He is tall.

## INS-06a  ·  INS_DUP / implausible / expect-edit
- observed: `We saw a a movie.`
- intended: `We saw a movie.`
- logZ -38.14, seed-spread 1.513 nats

  1. `0.758`  We saw a movie.  ←intended
  2. `0.120`  We saw a a movie.  ←observed
  3. `0.094`  We saw a move.

## INS-06b  ·  INS_DUP / plausible / expect-keep
- observed: `We saw a movie.`
- intended: `We saw a movie.`
- logZ -31.05, seed-spread 0.967 nats

  1. `0.863`  We saw a movie.  ←intended
  2. `0.132`  We saw a move.
  3. `0.003`  We saw a movie

## INS-07a  ·  INS_DUP / implausible / expect-edit
- observed: `The book on the the table.`
- intended: `The book on the table.`
- logZ -37.33, seed-spread 0.988 nats

  1. `0.349`  The book on the table.  ←intended
  2. `0.213`  The book on the the table.  ←observed
  3. `0.136`  The Book on the Table.

## INS-07b  ·  INS_DUP / plausible / expect-keep
- observed: `The book on the table.`
- intended: `The book on the table.`
- logZ -30.66, seed-spread 0.461 nats

  1. `0.476`  The Book on the Table.
  2. `0.352`  The book on the table.  ←intended
  3. `0.094`  The Book on The Table.

## INS-08a  ·  INS_DUP / implausible / expect-edit
- observed: `They walked walked home slowly.`
- intended: `They walked home slowly.`
- logZ -43.01, seed-spread 16.401 nats

  1. `0.425`  They walked walked home slowly.  ←observed
  2. `0.409`  They walked and walked home slowly.
  3. `0.155`  They walked home slowly.  ←intended

## INS-08b  ·  INS_DUP / plausible / expect-keep
- observed: `They walked home slowly.`
- intended: `They walked home slowly.`
- logZ -33.18, seed-spread 1.916 nats

  1. `0.994`  They walked home slowly.  ←intended
  2. `0.002`  They walked slowly.
  3. `0.001`  They talked home slowly.

## INS-09a  ·  INS_DUP / implausible / expect-edit
- observed: `The very very tall man left.`
- intended: `The very tall man left.`
- logZ -42.78, seed-spread 2.304 nats

  1. `0.921`  The very very tall man left.  ←observed
  2. `0.051`  The very tall man left.  ←intended
  3. `0.012`  The very very tall man.

## INS-09b  ·  INS_DUP / plausible / expect-keep
- observed: `The very tall man left.`
- intended: `The very tall man left.`
- logZ -37.06, seed-spread 1.844 nats

  1. `0.962`  The very tall man left.  ←intended
  2. `0.018`  The Very tall man left.
  3. `0.017`  The very tall man.

## DEL-to-05a  ·  DEL_TO / implausible / expect-edit
- observed: `I want go home.`
- intended: `I want to go home.`
- logZ -29.29, seed-spread 0.322 nats

  1. `0.942`  I want to go home.  ←intended
  2. `0.047`  I want go home.  ←observed
  3. `0.010`  I want to home.

## DEL-to-05b  ·  DEL_TO / plausible / expect-keep
- observed: `I want to go home.`
- intended: `I want to go home.`
- logZ -24.61, seed-spread 0.53 nats

  1. `0.993`  I want to go home.  ←intended
  2. `0.005`  I want to go Home.
  3. `0.001`  I want to go home

## DEL-of-01a  ·  DEL_OF / implausible / expect-edit
- observed: `This is one the best.`
- intended: `This is one of the best.`
- logZ -29.58, seed-spread 1.832 nats

  1. `0.689`  This is one of the best.  ←intended
  2. `0.255`  This is one the best.  ←observed
  3. `0.026`  This is the best.

## DEL-of-01b  ·  DEL_OF / plausible / expect-keep
- observed: `This is one of the best.`
- intended: `This is one of the best.`
- logZ -24.47, seed-spread 2.146 nats

  1. `0.981`  This is one of the best.  ←intended
  2. `0.019`  This is one of the best
  3. `0.000`  This is one of The Best

## DEL-a-01a  ·  DEL_A / implausible / expect-edit
- observed: `He is good man.`
- intended: `He is a good man.`
- logZ -26.99, seed-spread 0.411 nats

  1. `0.831`  He is good man.  ←observed
  2. `0.161`  He is a good man.  ←intended
  3. `0.005`  He is Good man.

## DEL-a-01b  ·  DEL_A / plausible / expect-keep
- observed: `He is a good man.`
- intended: `He is a good man.`
- logZ -25.20, seed-spread 0.394 nats

  1. `0.995`  He is a good man.  ←intended
  2. `0.002`  He is a Good man.
  3. `0.002`  He is a Good Man.

## DEL-a-02a  ·  DEL_A / implausible / expect-edit
- observed: `She lives in big house.`
- intended: `She lives in a big house.`
- logZ -32.55, seed-spread 1.575 nats

  1. `0.971`  She lives in big house.  ←observed
  2. `0.016`  She lives in the big house.
  3. `0.009`  She lives in a big house.  ←intended

## DEL-a-02b  ·  DEL_A / plausible / expect-keep
- observed: `She lives in a big house.`
- intended: `She lives in a big house.`
- logZ -32.70, seed-spread 0.621 nats

  1. `0.997`  She lives in a big house.  ←intended
  2. `0.002`  She lives in a big House.
  3. `0.001`  She lives in big house.

## DEL-the-01a  ·  DEL_THE / implausible / expect-edit
- observed: `We went to store.`
- intended: `We went to the store.`
- logZ -30.80, seed-spread 3.955 nats

  1. `0.668`  We went to store.  ←observed
  2. `0.166`  We went to Store.
  3. `0.075`  We want to store.

## DEL-the-01b  ·  DEL_THE / plausible / expect-keep
- observed: `We went to the store.`
- intended: `We went to the store.`
- logZ -30.24, seed-spread 0.724 nats

  1. `0.976`  We went to the store.  ←intended
  2. `0.018`  We went to the Store.
  3. `0.002`  We went to The Store.

## INS-to-01a  ·  INS_TO / implausible / expect-edit
- observed: `The mother gave the daughter to the candle.`
- intended: `The mother gave the daughter the candle.`
- logZ -45.78, seed-spread 2.896 nats

  1. `0.986`  The mother gave the daughter to the candle.  ←observed
  2. `0.009`  The mother gave the daughter to the candles.
  3. `0.003`  The mother gave the daughter to the handle.

## INS-to-01b  ·  INS_TO / plausible / expect-keep
- observed: `The mother gave the candle to the daughter.`
- intended: `The mother gave the candle to the daughter.`
- logZ -45.10, seed-spread 1.642 nats

  1. `0.978`  The mother gave the candle to the daughter.  ←intended
  2. `0.007`  The mother gave the handle to the daughter.
  3. `0.005`  The mother gave the candles to the daughter.

## INS-to-02a  ·  INS_TO / implausible / expect-edit
- observed: `The waiter served the customers to the soup.`
- intended: `The waiter served the customers the soup.`
- logZ -47.91, seed-spread 7.129 nats

  1. `0.959`  The waiter served the customers to the soup.  ←observed
  2. `0.013`  The waiter serves the customers to the soup.
  3. `0.010`  The waiter served the customer to the soup.

## INS-to-02b  ·  INS_TO / plausible / expect-keep
- observed: `The waiter served the soup to the customers.`
- intended: `The waiter served the soup to the customers.`
- logZ -46.58, seed-spread 13.484 nats

  1. `0.965`  The waiter served the soup to the customers.  ←intended
  2. `0.015`  The waiter served the soup to the customer.
  3. `0.010`  The waiter serves the soup to the customers.

## INS-to-03a  ·  INS_TO / implausible / expect-edit
- observed: `The teacher read the children to the story.`
- intended: `The teacher read the children the story.`
- logZ -46.39, seed-spread 1.676 nats

  1. `0.967`  The teacher read the children to the story.  ←observed
  2. `0.014`  The teacher read the children to the store.
  3. `0.010`  The teacher read the children to the Story.

## INS-to-03b  ·  INS_TO / plausible / expect-keep
- observed: `The teacher read the story to the children.`
- intended: `The teacher read the story to the children.`
- logZ -42.13, seed-spread 9.234 nats

  1. `0.967`  The teacher read the story to the children.  ←intended
  2. `0.024`  The Teacher read the story to the children.
  3. `0.006`  The teacher reads the story to the children.

## INS-to-04a  ·  INS_TO / implausible / expect-edit
- observed: `The judge awarded the winner to the trophy.`
- intended: `The judge awarded the winner the trophy.`
- logZ -43.97, seed-spread 1.512 nats

  1. `0.518`  The Judge awarded the winner to the trophy.
  2. `0.419`  The judge awarded the winner to the trophy.  ←observed
  3. `0.015`  The Judge awarded the winner to the Trophy.

## INS-to-04b  ·  INS_TO / plausible / expect-keep
- observed: `The judge awarded the trophy to the winner.`
- intended: `The judge awarded the trophy to the winner.`
- logZ -42.79, seed-spread 1.949 nats

  1. `0.473`  The Judge awarded the trophy to the winner.
  2. `0.394`  The judge awarded the trophy to the winner.  ←intended
  3. `0.081`  The judges awarded the trophy to the winner.
